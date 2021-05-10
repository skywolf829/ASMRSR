from utility_functions import str2bool, to_pixel_samples, PSNR, ssim, ssim3D, make_coord
from options import *
from datasets import LocalDataset
from models import load_model
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
    parser.add_argument('--bilinear',default="false",type=str2bool)

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

    use_bilin = args['bilinear']

    scale_factor_to_test = np.arange(1, 5)
    psnrs = []
    ssims = []
    with torch.no_grad():
        for i in range(len(scale_factor_to_test)):
            scale_factor = scale_factor_to_test[i]
            psnrs_this_scale = []
            ssims_this_scale = []
            for j in range(len(dataset)):
                hr = dataset[j].to(args['device']).unsqueeze(0)
                real_shape = hr.shape

                size = []
                for i in range(2, len(hr.shape)):
                    size.append(round(hr.shape[i]*(1/scale_factor)))
                lr = F.interpolate(hr, size=size, 
                        mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                        align_corners=False, recompute_scale_factor=False)

                if(not use_bilin):
                    if(model.upscaling_model.continuous):
                        coords = make_coord(hr.shape[2:], args['device'], flatten=False)
                        cell_sizes = torch.ones_like(coords)

                        for i in range(cell_sizes.shape[-1]):
                            cell_sizes[:,:,i] *= 2 / size[i]
                        
                        lr_upscaled = model(lr, coords, cell_sizes)
                        if(args['mode'] == "2D"):
                            lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                        else:                    
                            lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                        #lr_upscaled = torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0)
                    else:
                        lr_upscaled = model(lr)
                else:
                    lr_upscaled = F.interpolate(lr, size=hr.shape[2:],
                        align_corners=False, mode="bilinear" if args['mode'] == "2D" else "trilinear")
                
                lr_upscaled = F.interpolate(lr_upscaled, size=real_shape[2:], mode="bilinear", align_corners=False)
                p = PSNR(lr_upscaled, hr)
                psnrs_this_scale.append(p.item())
            psnrs.append(np.array(psnrs_this_scale).mean())
            print("Scale factor x%i, PSNR (dB): %0.02f" % (scale_factor, psnrs[-1]))
            