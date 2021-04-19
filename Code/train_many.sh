#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ASMRSR

python3 Code/main.py --save_name isomag2D_RDN5_64 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 5 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:0 &

python3 Code/main.py --save_name isomag2D_RDN15_64 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 15 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:1 &

python3 Code/main.py --save_name isomag2D_RDN25_64 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 25 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:2 &

python3 Code/main.py --save_name isomag2D_RDN5_128 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 5 --base_num_kernels 128 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:3 &

python3 Code/main.py --save_name isomag2D_RDN15_128 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 15 --base_num_kernels 128 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:4 &

python3 Code/main.py --save_name isomag2D_RDN25_128 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 25 --base_num_kernels 128 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:5 &

python3 Code/main.py --save_name isomag2D_RRDN5_64 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 5 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:6 &

python3 Code/main.py --save_name isomag2D_RRDN15_64 --train_distributed False --num_workers 8 \
--data_folder TrainingData/isomag2D --mode 2D --patch_size 1024 \
--training_patch_size 256 --num_blocks 15 --base_num_kernels 64 \
--x_resolution 1024 --y_resolution 1024 --epochs 1000 \
--scale_factor_end 4 --cropping_resolution 256 --device cuda:7 