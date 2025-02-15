# IterDepth
The official code of IterDepth: Iterative Residual Refinement for Outdoor Self-Supervised Multi-Frame Monocular Depth Estimation


[Cheng Feng](https://scholar.google.com/citations?hl=en&user=7DWAC44AAAAJ), Zhen Chen, Congxuan Zhang, Weiming Hu, Bing Li, and Feng Lu. – **TCSVT 2024**

[[Link to paper]](https://ieeexplore.ieee.org/document/10147244)

We introduce ***IterDepth***, an iterative residual refinement network to dense depth estimation.



## ✏️ 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
@article{DBLP:journals/tcsv/FengCZHLL24,
  author       = {Cheng Feng and
                  Zhen Chen and
                  Congxuan Zhang and
                  Weiming Hu and
                  Bing Li and
                  Feng Lu},
  title        = {IterDepth: Iterative Residual Refinement for Outdoor Self-Supervised
                  Multi-Frame Monocular Depth Estimation},
  journal      = {{IEEE} Trans. Circuits Syst. Video Technol.},
  volume       = {34},
  number       = {1},
  pages        = {329--341},
  year         = {2024},
  url          = {https://doi.org/10.1109/TCSVT.2023.3284479},
  doi          = {10.1109/TCSVT.2023.3284479},
  timestamp    = {Fri, 26 Jan 2024 07:57:10 +0100},
  biburl       = {https://dblp.org/rec/journals/tcsv/FengCZHLL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## 👀 Reproducing Paper Results

To recreate the results from our paper, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name>
    --batch_size 8
    --iters 6
    --png
```

Depending on the size of your GPU, you may need to set `--batch_size` to be lower than 8. Additionally you can train a high resolution model by adding `--height 320 --width 1024`.

For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)


## 💾 Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI MR (640x192)](https://drive.google.com/drive/folders/1fnsYm4U7lqPMPKK6qKcEQQc-Ho2t8dPm?usp=sharing)
* [KITTI HR (1024x320)](https://drive.google.com/drive/folders/1H9sJLAd-yIXWTtRP36fJeMpEXz3Wd-zA?usp=sharing)


To evaluate a model on KITTI, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --iters 6
    --eval_mono
    --png
```

If you want to evaluate a teacher network (i.e. the monocular network used for consistency loss), then add the flag `--eval_teacher`. This will 
load the weights of `mono_encoder.pth` and `mono_depth.pth`, which are provided for our KITTI models. 

In my experience, employing different software environments can yield varying evaluation results even when using the same weight file. The specific versions of the software utilized in this article are outlined below:
```latex
numpy                         1.21.5
opencv-python                 4.5.5.64
Pillow-SIMD                   9.0.0.post1
torch                         1.11.0
torchvision                   0.12.0
```




## 👩‍⚖️ Acknowledgement and License
The majority of the code for this project comes from [Manydepth](https://github.com/nianticlabs/manydepth). We appreciate the outstanding contributions Project has made to this field.

Meanwhile, the licensing of this project is the same as that of [Manydepth](https://github.com/nianticlabs/manydepth).
