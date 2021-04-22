from utility_functions import make_coord, str2bool, to_img, PSNR, to_pixel_samples, make_residual_weight_grid
from options import *
from datasets import LocalDataset
from models import load_model
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import random
import imageio
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default="2D",help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default="isomag2D",type=str,help='Folder to test on')
    parser.add_argument('--load_from',default="isomag2D",type=str,help='Model to load and test')
    parser.add_argument('--device',default="cuda:0",type=str,help='Device to evaluate on')
    parser.add_argument('--max_sf',default=4.0,type=float,help='Max SR factor to test')

    parser.add_argument('--increasing_size_test',default="false",type=str2bool,
        help='Gradually increase output size test')
    parser.add_argument('--single_sf_test',default="false",type=str2bool,
        help='Perform a single SF test')
    parser.add_argument('--histogram_test',default="false",type=str2bool,
        help='Perform histogram test on inputs/outputs/target')
    parser.add_argument('--patchwise_reconstruction_test',default="false",type=str2bool,
        help='Patchwise reconstruction for a full image')
    parser.add_argument('--interpolation_comparison',default="false",type=str2bool,
        help='Linear interoplation comparison')
    parser.add_argument('--feature_maps_test',default="false",type=str2bool,
        help='Save feature maps for some output')
    parser.add_argument('--increasing_sr_test',default="false",type=str2bool,
        help='Gradually increase SR factor test')

        
    parser.add_argument('--weight_grid_test',default="true",type=str2bool,
        help='Gradually increase SR factor test')
        

    args = vars(parser.parse_args())

    torch.manual_seed(0b10101010101010101010101010101010)
    random.seed(0b10101010101010101010101010101010)
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
            
    opt['cropping_resolution'] = 64
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
                align_corners=False)
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
                    cell_sizes[:,:,i] *= 2 / size[i]
                
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
                    align_corners=False, recompute_scale_factor=False)

            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
            [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            coords = make_coord(hr.shape[2:], args['device'], flatten=False)
            cell_sizes = torch.ones_like(coords)

            for i in range(cell_sizes.shape[-1]):
                cell_sizes[:,:,i] *= 2 / size[i]
            
            lr_upscaled = model(lr, coords, cell_sizes)
            if(args['mode'] == "2D"):
                lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
            else:                    
                lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
            print("PSNR: %0.02f" % PSNR(hr, lr_upscaled))
            sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
            error_im = torch.abs(lr_upscaled-hr)

            lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            error_np = error_im[0].permute(1, 2, 0).detach().cpu().numpy()

            full_im = np.append(lr_np, sr_np, axis=1)
            full_im = np.append(full_im, hr_np, axis=1)
            cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_test.jpg"), full_im)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_error.jpg"), error_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_sr.jpg"), sr_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_hr.jpg"), hr_np)
            
        if(args['histogram_test']):
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)
            size = []
            for i in range(2, len(hr.shape)):
                size.append(int(hr.shape[i]*(1/args['max_sf'])))
            lr = F.interpolate(hr, size=size, 
                    mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                    align_corners=False, recompute_scale_factor=False)

            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
            [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            coords = make_coord(hr.shape[2:], args['device'], flatten=False)
            cell_sizes = torch.ones_like(coords)

            for i in range(cell_sizes.shape[-1]):
                cell_sizes[:,:,i] *= 2 / size[i]
            
            lr_upscaled = model(lr, coords, cell_sizes)
            if(args['mode'] == "2D"):
                lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
            else:                    
                lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
            print("PSNR: %0.02f" % PSNR(hr, lr_upscaled))
            sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
            error_im = torch.abs(lr_upscaled-hr)

            lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            error_np = error_im[0].permute(1, 2, 0).detach().cpu().numpy()

            full_im = np.append(lr_np, sr_np, axis=1)
            full_im = np.append(full_im, hr_np, axis=1)
            cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_test.jpg"), full_im)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_error.jpg"), error_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_sr.jpg"), sr_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_hr.jpg"), hr_np)

            #plt.hist(lr_np.flatten(), 50, color="blue", histtype='step',
            #    density=True, fill=False, label="Low res input")
            plt.hist(sr_np.flatten(), 50, color="green", histtype='step',
                density=True, fill=False, label="SR output")
            plt.hist(hr_np.flatten(), 50, color="red", histtype='step',
                density=True, fill=False, label="High res target")
            plt.title("Histograms of each image")
            plt.legend(loc="upper left")
            plt.show()

        if(args['patchwise_reconstruction_test']):
            patch_size = opt['training_patch_size']
            print(patch_size)
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            
            hr_orig = hr.clone()
            print(hr.shape)
            hr = F.unfold(hr, patch_size, stride=patch_size)
            print(hr.shape)
            hr = hr.view(hr.shape[0], patch_size, 
                patch_size, -1).contiguous().permute(3, 0, 1, 2)
            print(hr.shape)
    
            size = []
            for i in range(2, len(hr.shape)):
                size.append(int(hr.shape[i]*(1/args['max_sf'])))
            
            coords = make_coord(hr.shape[2:], args['device'], flatten=False)
            cell_sizes = torch.ones_like(coords)
            for i in range(cell_sizes.shape[-1]):
                cell_sizes[:,:,i] *= 2 / size[i]
            #quit()
            psnrs = []
            output_lrs = []
            for i in range(hr.shape[0]):
                hr_im = torch.from_numpy(np.transpose(to_img(hr[i:i+1], args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)

                lr = F.interpolate(hr[i:i+1], size=size, 
                        mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                        align_corners=False, recompute_scale_factor=False)
                
                lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
                lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

                lr_upscaled = model(lr, coords, cell_sizes)
                if(args['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                output_lrs.append(lr_upscaled)
                p = PSNR(hr[i:i+1], lr_upscaled).item()
                print("PSNR: %0.02f" % p)
                psnrs.append(p)

                sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                    [2, 0, 1])[0:3]).unsqueeze(0)
                sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
                error_im = torch.abs(lr_upscaled-hr)

                lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
                error_np = error_im[0].permute(1, 2, 0).detach().cpu().numpy()

                full_im = np.append(lr_np, sr_np, axis=1)
                full_im = np.append(full_im, hr_np, axis=1)
                cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
                imageio.imwrite(os.path.join(output_folder, "Single_SF_test.jpg"), full_im)
                #imageio.imwrite(os.path.join(output_folder, "Single_SF_error.jpg"), error_np)
                imageio.imwrite(os.path.join(output_folder, "Single_SF_sr.jpg"), sr_np)
                imageio.imwrite(os.path.join(output_folder, "Single_SF_hr.jpg"), hr_np)
            print("Average PSNR: %0.02f" % np.array(psnrs).mean())

            output_lrs = torch.cat(output_lrs, dim=0)
            print(output_lrs.shape)
            output_lrs = output_lrs.permute(1, 2, 3, 0).view(1, 
                patch_size*patch_size, -1)
            print(output_lrs.shape)
            output_lrs = F.fold(output_lrs, output_size=hr_orig.shape[2:], 
                kernel_size=patch_size, stride=patch_size)
            print(output_lrs.shape)
            print("Final PSNR: %0.02f" % PSNR(hr_orig, output_lrs).item())

            sr_im = torch.from_numpy(np.transpose(to_img(output_lrs, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            hr_im = torch.from_numpy(np.transpose(to_img(hr_orig, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            imageio.imwrite(os.path.join(output_folder, "Single_SF_sr.jpg"), sr_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_hr.jpg"), hr_np)

        if(args['interpolation_comparison']):
            rand_dataset_item = random.randint(0, len(dataset)-1)
            hr = dataset[rand_dataset_item].unsqueeze(0).to(args['device'])
            hr_im = torch.from_numpy(np.transpose(to_img(hr, args['mode']), 
                            [2, 0, 1])[0:3]).unsqueeze(0)
            size = []
            for i in range(2, len(hr.shape)):
                size.append(int(hr.shape[i]*(1/args['max_sf'])))
            lr = F.interpolate(hr, size=size, 
                    mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                    align_corners=False, recompute_scale_factor=False)

            lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
            [2, 0, 1])[0:3]).unsqueeze(0)
            lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

            lr_interp = F.interpolate(lr, size=hr.shape[2:], 
                mode='bilinear' if opt['mode'] == "2D" else "trilinear",
                align_corners=False, recompute_scale_factor=False)
            
            coords = make_coord(hr.shape[2:], args['device'], flatten=False)
            cell_sizes = torch.ones_like(coords)

            for i in range(cell_sizes.shape[-1]):
                cell_sizes[:,:,i] *= 2 / size[i]
            
            lr_upscaled = model(lr, coords, cell_sizes)
            if(args['mode'] == "2D"):
                lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
            else:                    
                lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
            print(hr.shape)
            print(lr_interp.shape)
            print(lr_upscaled.shape)
            print("PSNR from SR: %0.02f" % PSNR(hr, lr_upscaled))
            print("PSNR from interp: %0.02f" % PSNR(hr, lr_interp))

            sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            sr_im = F.interpolate(sr_im, size=hr_im.shape[2:], mode='nearest')
            
            error_im = torch.abs(lr_upscaled-hr)
            
            interp_im = torch.from_numpy(np.transpose(to_img(lr_interp, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
            interp_im = F.interpolate(interp_im, size=hr_im.shape[2:], mode='nearest')
            

            lr_np = lr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            sr_np = sr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            interp_np = interp_im[0].permute(1, 2, 0).detach().cpu().numpy()
            hr_np = hr_im[0].permute(1, 2, 0).detach().cpu().numpy()
            error_np = error_im[0].permute(1, 2, 0).detach().cpu().numpy()

            full_im = np.append(lr_np, interp_np, axis=1)
            full_im = np.append(full_im, sr_np, axis=1)
            full_im = np.append(full_im, hr_np, axis=1)
            cv2.putText(full_im, "x%0.01f" % (hr.shape[2]/lr.shape[2]), (lr_np.shape[0], 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 0), 1)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_test.jpg"), full_im)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_error.jpg"), error_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_interp.jpg"), interp_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_sr.jpg"), sr_np)
            imageio.imwrite(os.path.join(output_folder, "Single_SF_hr.jpg"), hr_np)

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
                    align_corners=False)

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
                    align_corners=False)
                lr_im = torch.from_numpy(np.transpose(to_img(lr, args['mode']), 
                [2, 0, 1])[0:3]).unsqueeze(0)
                lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')

                coords = make_coord(hr.shape[2:], args['device'], flatten=False)
                cell_sizes = torch.ones_like(coords)

                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,:,i] *= 2 / size[i]
                
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

        if(args['weight_grid_test']):
            ex_hr = torch.rand([1, 1, opt['cropping_resolution'], opt['cropping_resolution']], 
                device=opt['device'], dtype=torch.float32)
            hr_coords, hr_pix = to_pixel_samples(ex_hr, flatten=False)
            
            up_size = 512
            
            hr_coords_color = hr_coords.clone()
            hr_coords_color += 1
            hr_coords_color *= (up_size/2)
            hr_coords_color = hr_coords_color.type(torch.LongTensor)

            ims = []
            for s in range(opt['cropping_resolution'], 
                int(opt['cropping_resolution']/args['max_sf'])-1, -1):
                print(s)
                ex_lr = F.interpolate(ex_hr, size=[s, s],
                        mode = "bilinear" if opt['mode'] == "2D" else "trilinear",
                        align_corners=False, recompute_scale_factor=False)

                rel_coord = make_residual_weight_grid(ex_lr, hr_coords, opt['mode']).expand(-1, 3, -1, -1).contiguous()
                
                rel_coord = F.interpolate(rel_coord, size=[up_size, up_size], mode='nearest')

                lr_coord = make_coord(ex_lr.shape[2:], device=ex_lr.device,
                    flatten=False)
                lr_coord += 1
                lr_coord *= (up_size/2)
                lr_coord = lr_coord.type(torch.LongTensor)

                for i in range(lr_coord.shape[0]):
                    for j in range(lr_coord.shape[1]):
                        x = lr_coord[i,j,0].item()
                        y = lr_coord[i,j,1].item()
                        rel_coord[0, 0, x-1:x+2, y-1:y+2] = 1.0
                        rel_coord[0, 1, x-1:x+2, y-1:y+2] = 0.0
                        rel_coord[0, 2, x-1:x+2, y-1:y+2] = 0.0
                
                for i in range(hr_coords_color.shape[0]):
                    for j in range(hr_coords_color.shape[1]):
                        x = hr_coords_color[i,j,0].item()
                        y = hr_coords_color[i,j,1].item()
                        rel_coord[0, 0, x-1:x+2, y-1:y+2] = 0.0
                        rel_coord[0, 1, x-1:x+2, y-1:y+2] = 1.0
                        rel_coord[0, 2, x-1:x+2, y-1:y+2] = 0.0
                
                
                im = rel_coord[0].permute(1, 2, 0).cpu().detach().numpy() * 255
                im = im.astype(np.uint8)
                print(im.shape)
                for k in range(10):
                    ims.append(im)
            imageio.mimwrite(os.path.join(output_folder, "ResidualWeights.gif"), ims)
            imageio.mimwrite(os.path.join(output_folder, "ResidualWeights.mp4"), ims)