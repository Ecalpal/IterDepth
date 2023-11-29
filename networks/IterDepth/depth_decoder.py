# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import ConvBlock, Conv3x3, upsample
from networks.IterDepth.update import BasicUpdateBlock, SmallUpdateBlock


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class DepthDecoder_GRU(nn.Module):
    def __init__(self, iters=6, res=18):
        super(DepthDecoder_GRU, self).__init__()

        self.iters = iters

        if res==18:
            self.update_block = BasicUpdateBlock(in_e=128, in_c=256)
            self.conv_e1 = ConvBlock(512, 256)
            self.conv_e2 = ConvBlock(256, 128)
            self.conv_c1 = ConvBlock(512, 256)
            self.conv_c21 = ConvBlock(128, 128)
            self.conv_c22 = ConvBlock(128, 128)
        elif res==50:
            self.update_block = BasicUpdateBlock(in_e=512, in_c=1024)
            self.conv_e1 = ConvBlock(2048, 1024)
            self.conv_e2 = ConvBlock(1024, 512)
            self.conv_c1 = ConvBlock(2048, 1024)
            self.conv_c21 = ConvBlock(512, 512)
            self.conv_c22 = ConvBlock(512, 512)

        self.sigmoid = nn.Sigmoid()

    def upsample_depth(self, depth, mask):
        """ Upsample depth map [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        N, C, H, W = depth.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_depth = nn.functional.unfold(depth, [3,3], padding=1)
        up_depth = up_depth.view(N, C, 9, 1, 1, H, W)

        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(N, C, 8*H, 8*W)

    def forward(self, encoder_fea, context_fea, test_mode=False, depth_init=None,gt=None):
        self.outputs = {}

        e_fea = self.conv_e1(encoder_fea[-1])
        e_fea = upsample(e_fea)
        e_fea += encoder_fea[-2]

        e_fea = self.conv_e2(e_fea)
        e_fea = upsample(e_fea)
        e_fea += encoder_fea[-3]

        c_fea = self.conv_c1(context_fea[-1])
        c_fea = upsample(c_fea)
        c_fea += context_fea[-2]
        net, inp = torch.split(c_fea, [c_fea.size()[1]//2, c_fea.size()[1]//2], dim=1)

        net = self.conv_c21(net)
        net = upsample(net)
        net += context_fea[-3]

        inp = self.conv_c22(inp)
        inp = upsample(inp)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        b, c, h, w = e_fea.size()

        if depth_init == None:
            depth = torch.zeros([b, 1, h, w], requires_grad=True).to(e_fea.device)
        else:
            depth = depth_init

        # import matplotlib as mpl
        # import matplotlib.cm as cm
        # from layers import disp_to_depth
        # import cv2

        for itr in range(self.iters):

            net, up_mask, delta_depth = self.update_block(net, inp, e_fea, depth)

            depth = self.sigmoid(depth + delta_depth)

            # upsample predictions
            if up_mask is None:
                # depth_up = updepth8(coords1 - coords0)
                print("TODO")
            else:
                depth_up = self.upsample_depth(depth, up_mask)
            
            self.outputs["disp_lr"] = depth
            self.outputs[("disp", 0, itr)] = depth_up

        #     pred_disp, pred_depth = disp_to_depth(depth_up, 0.1, 100)
        #     pred_disp = pred_disp.cpu()[:, 0].numpy()
        #     normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=np.percentile(pred_disp, 95))
        #     mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        #     colormapped_im = (mapper.to_rgba(pred_disp[0])[:, :, :3] * 255).astype(np.uint8)
        #     cv2.imshow(f"iter_{itr}", colormapped_im[...,::-1])

        #     if itr == 0:
        #         prv = pred_disp
        #     else:
        #         diff = pred_disp - prv
        #         prv = pred_disp
        #         diff_rgb = np.zeros((diff.shape[1],diff.shape[2],3)).astype(np.uint8)

        #         scale = diff.max() if diff.max() > diff.min()*-1 else diff.min()*-1
        #         diff = diff[0] / scale * 255.0 * 3
        #         diff[diff> 255]=255
        #         diff[diff< -255]=-255

        #         diff_up   = diff.copy()
        #         diff_down = diff.copy()

        #         diff_up[diff_up<0] = 0
        #         diff_up = diff_up.astype(np.uint8)

        #         diff_down[diff_down>0]=0
        #         diff_down = diff_down * -1
        #         diff_down = diff_down.astype(np.uint8)

        #         diff_rgb[:,:,0] = diff_up
        #         diff_rgb[:,:,2] = diff_down

        #         cv2.imshow(f"diff_{itr}", diff_rgb)
            
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #     gt_vis = gt[0][0].cpu().numpy()
        #     gt_depth_vis = cv2.dilate(gt_vis, kernel)
        #     pred_depth = cv2.resize(pred_depth.cpu().numpy()[0][0], (gt_depth_vis.shape[1], gt_depth_vis.shape[0]))
        #     ratio = np.median(gt_vis[gt_vis>0]) / np.median(pred_depth[gt_vis>0])
        #     pred_depth = pred_depth*ratio
        #     pred_depth[pred_depth<1e-3] = 1e-3
        #     pred_depth[pred_depth>80] = 80
        #     mask_vis = gt_depth_vis > 0
        #     gt_depth_vis[gt_depth_vis<1e-3] = 1e-3
        #     gt_depth_vis[gt_depth_vis>80] = 80
            

        #     max_error = 10
        #     error_map = abs(gt_depth_vis-pred_depth)[None].transpose(1,2,0) / max_error
        #     error_map [error_map > 1] = 1
        #     error_map = (error_map*255.0).astype(np.uint8)
        #     error_map[:,:,0] = error_map[:,:,0] * mask_vis
        #     error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)  # 在原图上应用不同的颜色模式
        #     error_map[:,:,0] = error_map[:,:,0] * mask_vis

        #     cv2.imshow(f"error_{itr}", error_map)


        # cv2.waitKey(0)
            
        if test_mode:
            return depth_up, depth, up_mask
        else:
            return self.outputs


import networks.IterDepth as networks
class Fullmodel(nn.Module):
    def __init__(self, h, w, num_layers=18, iters=6):
        super(Fullmodel, self).__init__()
        self.encoder = networks.ResnetEncoderMatching(
            num_layers=num_layers, pretrained=False,
            input_height=h, input_width=w,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning='linear', num_depth_bins=96)
        self.encoder.cuda()

        self.context = networks.ResnetEncoder_context(num_layers=num_layers, pretrained=False)
        self.context.cuda()

        self.depth = networks.DepthDecoder_GRU(iters=iters, res=num_layers)
        self.depth.cuda()

    def forward(self, input_color, lookup_frames, relative_poses, K, invK, test_mode=False, depth_init=None):
        encoder_output, lowest_cost, costvol = self.encoder(input_color, lookup_frames,
                                                            relative_poses,
                                                            K,
                                                            invK,
                                                            0, 80)
        context_output = self.context(input_color)
        output, out_lr = self.depth(encoder_output, context_output, test_mode=True)

        return output
