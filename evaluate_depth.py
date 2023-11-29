# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
import tqdm
from torch.utils.data import DataLoader

import datasets
import networks.IterDepth as networks
from layers import disp_to_depth, transformation_from_parameters
from options import MonodepthOptions
from utils import readlines

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            encoder_context_path = os.path.join(opt.load_weights_folder, "encoder_context.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_class = networks.ResnetEncoderMatching

        encoder_dict = torch.load(encoder_path)
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False)

        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False, use_depth_hints=False,
                                               img_ext=".png" if opt.png else ".jpg")
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # setup models
        if opt.eval_teacher:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False)
        else:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins)
            pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
            pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

            pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
            pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                            num_frames_to_predict_for=2)

            pose_enc.load_state_dict(pose_enc_dict, strict=True)
            pose_dec.load_state_dict(pose_dec_dict, strict=True)

            min_depth_bin = encoder_dict.get('min_depth_bin')
            max_depth_bin = encoder_dict.get('max_depth_bin')

            pose_enc.eval()
            pose_dec.eval()

            if torch.cuda.is_available():
                pose_enc.cuda()
                pose_dec.cuda()

        encoder = encoder_class(**encoder_opts)
        encoder_context = networks.ResnetEncoder_context(opt.num_layers, pretrained=False)
        depth_decoder = networks.DepthDecoder_GRU(opt.iters, opt.num_layers)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        encoder_context.load_state_dict(torch.load(encoder_context_path))
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        encoder_context.eval()
        depth_decoder.eval()

        if torch.cuda.is_available():
            encoder.cuda()
            encoder_context.cuda()
            depth_decoder.cuda()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        
        if opt.save_pred_disps:
            if opt.zero_cost_volume:
                tag = "zero_cv"
            elif opt.eval_teacher:
                tag = "teacher"
            else:
                tag = "multi"
            
            output_dir = f"./test_result/self"
            output_dir_disp_vis = f"{output_dir}/{tag}/disp_vis"
            os.makedirs(output_dir_disp_vis, exist_ok=True)
            output_dir_depth_vis = f"{output_dir}/{tag}/depth_vis"
            os.makedirs(output_dir_depth_vis, exist_ok=True)
            output_dir_error_vis = f"./{output_dir}/{tag}/error_vis"
            os.makedirs(output_dir_error_vis, exist_ok=True)
            output_dir_gt_vis = f"./{output_dir}/{tag}/gt_vis"
            os.makedirs(output_dir_gt_vis, exist_ok=True)

            save_name = []

            with open("./splits/eigen/test_files.txt","r") as f:
                lines = f.readlines()
                for line in lines:
                    folder, idx, d = line[:-1].split(' ')
                    lr = "image_02" if d=="l" else "image_03"
                    save_name.append(f"{folder.split('/')[-1]}_{lr}_{idx}")
                f.close()

        # do inference
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):


                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                if opt.eval_teacher:
                    output = encoder(input_color)
                    output = depth_decoder(output)
                else:
                    if opt.static_camera:
                        for f_i in frames_to_load:
                            data["color", f_i, 0] = data[('color', 0, 0)]

                    # predict poses
                    pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
                    if torch.cuda.is_available():
                        pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in frames_to_load[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)

                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                        data[('relative_pose', fi)] = pose

                    lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
                    lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                    relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                    relative_poses = torch.stack(relative_poses, 1)

                    K = data[('K', 2)]  # quarter resolution for matching
                    invK = data[('inv_K', 2)]

                    if torch.cuda.is_available():
                        lookup_frames = lookup_frames.cuda()
                        relative_poses = relative_poses.cuda()
                        K = K.cuda()
                        invK = invK.cuda()

                    if opt.zero_cost_volume:
                        relative_poses *= 0

                    if opt.post_process:
                        raise NotImplementedError

                    encoder_output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                           relative_poses,
                                                           K,
                                                           invK,
                                                           min_depth_bin, max_depth_bin)
                    context_output = encoder_context(input_color)

                    output, _, _ = depth_decoder(encoder_output, context_output, test_mode=True, gt=data['depth_gt'])

                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        if opt.zero_cost_volume:
            tag = "zero_cv"
        elif opt.eval_teacher:
            tag = "teacher"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_split.npy".format(tag, opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

        for vis_idx in range(len(pred_disps)):
            disp_resized = pred_disps[vis_idx]
            normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=np.percentile(disp_resized, 95))
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            im.save(f"{output_dir_disp_vis}/{save_name[vis_idx]}_disp.jpg")


    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    if opt.eval_split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    depth_vis = []
    depth_masked_vis = []
    gt_vis = []
    error_vis = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = gt_depth > 0

        

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gt_depth_vis = cv2.dilate(gt_depth, kernel)
        # gt_depth_vis = cv2.dilate(gt_depth_vis, kernel)
        # gt_depth_vis = cv2.dilate(gt_depth_vis, kernel)

        mask_vis = gt_depth_vis > 0

        pred_depth_vis = pred_depth
        pred_depth_vis_masked = pred_depth * mask_vis

        # gt_depth_vis = gt_depth * mask

        pred_depth_vis *= opt.pred_depth_scale_factor
        pred_depth_vis_masked *= opt.pred_depth_scale_factor


        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor


        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            pred_depth_vis *= ratio
            pred_depth_vis_masked *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth_vis[pred_depth_vis < MIN_DEPTH] = MIN_DEPTH
        pred_depth_vis[pred_depth_vis > MAX_DEPTH] = MAX_DEPTH

        pred_depth_vis_masked[pred_depth_vis_masked < MIN_DEPTH] = MIN_DEPTH
        pred_depth_vis_masked[pred_depth_vis_masked > MAX_DEPTH] = MAX_DEPTH

        max_error = 10
        error_map = abs(gt_depth_vis-pred_depth_vis)[None].transpose(1,2,0) / max_error
        error_map [error_map > 1] = 1
        error_map = (error_map*255.0).astype(np.uint8)
        error_map[:,:,0] = error_map[:,:,0] * mask_vis
        error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)  # 在原图上应用不同的颜色模式
        error_map[:,:,0] = error_map[:,:,0] * mask_vis
        error_vis.append(error_map)

        gt_vis.append(gt_depth_vis)

        errors.append(compute_errors(gt_depth, pred_depth))

        if opt.save_pred_disps:
            depth_vis.append(pred_depth_vis)
            depth_masked_vis.append(pred_depth_vis_masked)

    if opt.save_pred_disps:
        print("saving errors")
        if opt.zero_cost_volume:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))

        normalizer = mpl.colors.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

        for vis_idx in range(len(depth_vis)):
            colormapped_im = (mapper.to_rgba(depth_vis[vis_idx])[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            im.save(f"{output_dir_depth_vis}/{save_name[vis_idx]}_depth.jpg")

            colormapped_im = (mapper.to_rgba(depth_masked_vis[vis_idx])[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            im.save(f"{output_dir_depth_vis}/{save_name[vis_idx]}_depth_masked.jpg")

            colormapped_im = (mapper.to_rgba(gt_vis[vis_idx])[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            im.save(f"{output_dir_gt_vis}/{save_name[vis_idx]}_gt_depth.jpg")

            im.save(f"{output_dir_gt_vis}/{save_name[vis_idx]}_gt_depth.jpg")


            cv2.imwrite(f"{output_dir_error_vis}/{save_name[vis_idx]}_error.jpg", error_vis[vis_idx])

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
