from utility_functions import make_coord, str2bool, to_img
from options import *
from datasets import LocalDataset
from models import load_model, LIIF_Generator
from train import Trainer
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import random
import imageio
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default="2D",help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default="isomag2D",type=str,help='Folder to test on')
    parser.add_argument('--load_from',default="isomag2D",type=str,help='Model to load and test')
    parser.add_argument('--device',default="cuda:0",type=str,help='Device to evaluate on')
    parser.add_argument('--max_sf',default=8.0,type=float,help='Max SR factor to test')

    parser.add_argument('--increasing_size_test',default="false",type=str2bool,
        help='Gradually increase output size test')
    parser.add_argument('--single_sf_test',default="true",type=str2bool,
        help='Perform a single SF test')
    parser.add_argument('--feature_maps_test',default="false",type=str2bool,
        help='Save feature maps for some output')
    parser.add_argument('--increasing_sr_test',default="false",type=str2bool,
        help='Gradually increase SR factor test')

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
            
    opt['cropping_resolution'] = 32
    opt['data_folder'] = os.path.join(input_folder, args['data_folder'])
    model = load_model(opt,args["device"]).to(args['device'])
    dataset = LocalDataset(opt)

    with torch.no_grad():
        if(args['increasing_size_test']):
            img_sequence = []
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)

            lr = F.interpolate(hr, scale_factor = (1/args['max_sf']), 
                mode='bilinear' if args['mode'] == "2D" else "trilinear",
                align_corners=True)
            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            for i in range(15):
                img_sequence.append(lr_im[0].permute(1, 2, 0).detach().cpu().numpy())
            
            sizes = []
            for i in range(2, len(hr.shape)):
                s = [lr.shape[i], hr.shape[i]]
                sizes.append(np.arange(s[0], s[1], (s[1]-s[0])/100))

            for i in range(len(sizes[0])):
                size = []
                for j in range(len(sizes)):
                    size.append(sizes[j][i])

                coords = make_coord(size, args['device'], flatten=False)
                cell_sizes = torch.ones_like(coords)

                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,i] *= 2 / size[i]
                
                lr_upscaled = model(lr, coords, cell_sizes)
                if(args['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                    [2, 0, 1])[0:3]).unsqueeze(0)
                sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
                img_sequence.append(sr_im[0].permute(1, 2, 0).detach().cpu().numpy())
            imageio.mimwrite(os.path.join(output_folder, "IncreasingSizeTest.gif"), img_sequence)

        if(args['single_sf_test']):
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)
            size = []
            for i in range(2, len(hr.shape)):
                size.append(int(hr.shape[i]*(1/args['max_sf'])))
            lr = F.interpolate(hr, size=size, 
                    mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                    align_corners=True)

            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
            [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            coords = make_coord(hr.shape[2:], args['device'], flatten=False)
            cell_sizes = torch.ones_like(coords)

            for i in range(cell_sizes.shape[-1]):
                cell_sizes[:,i] *= 2 / size[i]
            
            lr_upscaled = model(lr, coords, cell_sizes)
            if(args['mode'] == "2D"):
                lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
            else:                    
                lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
            sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
            lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            full_im = np.append(lr_np, sr_np, axis=1)
            full_im = np.append(full_im, hr_np, axis=1)
            cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_test.jpg"), full_im)

        if(args['feature_maps_test']):
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)
            size = []
            for i in range(2, len(hr.shape)):
                size.append(int(hr.shape[i]*(1/args['max_sf'])))
            lr = F.interpolate(hr, size=size, 
                    mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                    align_corners=True)

            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
            [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            features = model.feature_extractor(lr)
            for k in range(features.shape[1]):
                f_k = torch.from_numpy(np.transpose(to_img(features[0,k].unsqueeze(0).unsqueeze(0), opt['mode']), 
                    [2, 0, 1])[0:3]).unsqueeze(0) 
                f_k = F.interpolate(f_k, mode="nearest", size=hr_im.shape[2:])
                imageio.imwrite(os.path.join(output_folder, "feature_%i.jpg" % k), 
                    f_k[0].permute(1, 2, 0).detach().cpu().numpy())

            lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            imageio.imwrite(os.path.join(output_folder, "feature_input.jpg"), lr_np)

        if(args['increasing_sr_test']):
            img_sequence = []
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)
            
            sizes = []
            for i in range(2, len(hr.shape)):
                s = [hr.shape[i]*(1/args['max_sf']), hr.shape[i]]
                sizes.append(np.arange(s[0], s[1], 1, dtype=int))
            

            for i in range(len(sizes[0])):
                size = []
                for j in range(len(sizes)):
                    size.append(sizes[j][i])
                lr = F.interpolate(hr, size=size, 
                    mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                    align_corners=True)
                lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
                lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

                coords = make_coord(hr.shape[2:], args['device'], flatten=False)
                cell_sizes = torch.ones_like(coords)

                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,i] *= 2 / size[i]
                
                lr_upscaled = model(lr, coords, cell_sizes)
                if(args['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                    [2, 0, 1])[0:3]).unsqueeze(0)
                sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')

                
                lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                full_im = np.append(lr_np, sr_np, axis=1)
                full_im = np.append(full_im, hr_np, axis=1)
                cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
                img_sequence.append(full_im)
            imageio.mimwrite(os.path.join(output_folder, "IncreasingSRTest.gif"), img_sequence)