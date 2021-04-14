from utility_functions import str2bool, to_pixel_samples, PSNR, ssim, ssim3D
from options import *
from datasets import LocalDataset
from models import load_model, LIIF_Generator
from train import Trainer
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default="2D",help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default="isomag2D",type=str,help='Folder to test on')
    parser.add_argument('--load_from',default="Temp",type=str,help='Model to load and test')
    parser.add_argument('--device',default="cuda:0",type=str,help='Device to evaluate on')

    args = vars(parser.parse_args())

    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TestingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    opt = load_options(os.path.join(save_folder, args["load_from"]))
    opt["device"] = args["device"]
    opt["save_name"] = args["load_from"]
    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]
    opt['cropping_resolution'] = -1
    opt['data_folder'] = os.path.join(input_folder, args['data_folder'])
    model = load_model(opt,args["device"]).to(args['device'])
    dataset = LocalDataset(opt)

    scale_factor_to_test = np.arange(2, 30)
    psnrs = []
    ssims = []
    with torch.no_grad():
        for i in range(len(scale_factor_to_test)):
            scale_factor = scale_factor_to_test[i]
            psnrs_this_scale = []
            ssims_this_scale = []
            for j in range(len(dataset)):
                real_hr = dataset[j].to(args['device']).unsqueeze(0)
                real_shape = real_hr.shape

                real_lr = F.interpolate(real_hr, scale_factor=(1/scale_factor),
                    mode = "bilinear" if args['mode'] == "2D" else "trilinear",
                    align_corners=True, recompute_scale_factor=False)
            
                hr_coords, real_hr = to_pixel_samples(real_hr, flatten=False)
                cell_sizes = torch.ones_like(hr_coords)

                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,i] *= 2 / real_shape[2+i]
                
                lr_upscaled = model(real_lr, hr_coords, cell_sizes)
                if(args['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                

                p = PSNR(torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0), real_hr)
                #if(args['mode'] == "2D"):
                #    s = ssim()
                psnrs_this_scale.append(p.item())
            psnrs.append(np.array(psnrs_this_scale).mean())
            print("Scale factor x%i, PSNR (dB): %0.02f" % (scale_factor, psnrs[-1]))
            