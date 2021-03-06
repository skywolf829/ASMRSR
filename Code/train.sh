#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ASMRSR

python3 Code/main.py --save_name isomag2D_RDN5_64kernels_MFFN_temporal \
--mode 2DTV \
--train_distributed True --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 48 --cropping_resolution 48 \
--time_cropping_resolution 150 \
--feat_model RDN --upscale_model MFFN_temporal \
--num_blocks 5 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_start 1 --scale_factor_end 10 --residual_weighing false