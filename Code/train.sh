#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ASMRSR

python3 Code/main.py --save_name isomag2D_RDN15_patch256 --train_distributed True --num_workers 2 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 15 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256