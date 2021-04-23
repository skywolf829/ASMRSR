from utility_functions import to_pixel_samples, to_img, PSNR, make_coord, make_residual_weight_grid
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
import os
from models import save_model
import numpy as np

class Trainer():
    def __init__(self, opt):
        self.opt = opt

    def train_distributed(self, rank, model, opt, dataset):
        opt['device'] = "cuda:" + str(rank)
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size = self.opt['num_nodes'] * self.opt['gpus_per_node'],                              
            rank=rank                                               
        )
        model = model.to(rank)
        model = DDP(model, device_ids=[rank]) 

        model_optim = optim.Adam(model.parameters(), lr=self.opt["g_lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
            milestones=[200, 400, 600, 800],gamma=self.opt['gamma'])

        if(rank == 0):
            writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
        
        start_time = time.time()

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, 
            num_replicas=opt['num_nodes']*opt['gpus_per_node'], rank=rank)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=opt["num_workers"],
            pin_memory=True,
            sampler=train_sampler
        )
        L1loss = nn.L1Loss().to(opt["device"])
        step = 0
        for epoch in range(opt['epoch_number'], opt['epochs']):
            opt["epoch_number"] = epoch
            for batch_num, real_hr in enumerate(dataloader):
                model.zero_grad()
                real_hr = real_hr.to(self.opt['device'])
                if(rank == 0):
                    hr_im = torch.from_numpy(np.transpose(to_img(real_hr, self.opt['mode']), 
                        [2, 0, 1])[0:3]).unsqueeze(0)
                real_shape = real_hr.shape
                #print(real_hr.dtype)
                #print("Full shape : " + str(real_hr.shape))
                if(epoch > 100):
                    scale_factor = torch.rand([1], device=real_hr.device, dtype=real_hr.dtype) * \
                        (self.opt['scale_factor_end'] - self.opt['scale_factor_start']) + \
                            self.opt['scale_factor_start']
                else:
                    scale_factor = 1.0
                #scale_factor = 1
                #print("Scale factor: " + str(scale_factor))
                real_lr = F.interpolate(real_hr, scale_factor=(1/scale_factor),
                    mode = "bilinear" if self.opt['mode'] == "2D" else "trilinear",
                    align_corners=False, recompute_scale_factor=False)
                if(rank == 0):
                    lr_im = torch.from_numpy(np.transpose(to_img(real_lr, self.opt['mode']), 
                        [2, 0, 1])[0:3]).unsqueeze(0)
                    lr_im = F.interpolate(lr_im, size=hr_im.shape[2:], mode='nearest')
                #print("LR shape : " + str(real_lr.shape))

                #lr_upscaled = model(real_lr, list(real_hr.shape[2:]))
                hr_coords, real_hr = to_pixel_samples(real_hr, flatten=False)
                cell_sizes = torch.ones_like(hr_coords)
                #print("Cell sizes : " + str(cell_sizes.shape))
                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,:,i] *= 2 / real_shape[2+i]
                
                lr_upscaled = model(real_lr, hr_coords, cell_sizes)
                if(self.opt['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
                if(rank == 0):
                    sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, 
                        self.opt['mode']),[2, 0, 1])[0:3]).unsqueeze(0)

                if(self.opt['residual_weighing']):
                    lr_interp = F.grid_sample(real_lr, hr_coords.flip(-1).unsqueeze(0), 
                        mode = "bilinear" if self.opt['mode'] == "2D" else "trilinear",
                        align_corners=False)
                    
                    rel_coord = make_residual_weight_grid(real_lr, hr_coords, self.opt['mode'])
                    #print(rel_coord)
                    lr_interp *= (1-rel_coord)
                    lr_upscaled *= rel_coord
                    lr_upscaled += lr_interp.detach()

                L1 = L1loss(torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0), real_hr)
                L1.backward()
                model_optim.step()
                optim_scheduler.step()

                psnr = PSNR(torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0), real_hr)
                if(rank == 0 and step % self.opt['save_every'] == 0):
                    print("Epoch %i batch %i, sf: x%0.02f, L1: %0.04f, PSNR (dB): %0.02f" % \
                    (epoch, batch_num, scale_factor, L1.item(), psnr.item()))                    
                    writer.add_scalar('L1', L1.item(), step)
                    writer.add_images("LR, SR, HR", torch.cat([lr_im, sr_im, hr_im]), global_step=step)
                step += 1
            
            if(rank == 0 and epoch % self.opt['save_every'] == 0):
                save_model(model, self.opt)
                print("Saved model")

        end_time = time.time()
        total_time = start_time - end_time
        if(rank == 0):
            print("Time to train: " + str(total_time))
            save_model(model, self.opt)
            print("Saved model")

    def train_single(self, model, dataset):
        model = model.to(self.opt['device'])

        model_optim = optim.Adam(model.parameters(), lr=self.opt["g_lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
            milestones=[200, 400, 600, 800],gamma=self.opt['gamma'])

        writer = SummaryWriter(os.path.join('tensorboard',self.opt['save_name']))

        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=True,
            num_workers=self.opt["num_workers"],
            pin_memory=True
        )
        L1loss = nn.L1Loss().to(self.opt["device"])
        step = 0
        for epoch in range(self.opt['epoch_number'], self.opt['epochs']):
            self.opt["epoch_number"] = epoch
            for batch_num, real_hr in enumerate(dataloader):
                model.zero_grad()
                real_hr = real_hr.to(self.opt['device'])
                hr_im = torch.from_numpy(np.transpose(to_img(real_hr, self.opt['mode']), 
                        [2, 0, 1])[0:3]).unsqueeze(0)
                real_shape = real_hr.shape
                if(epoch <= 100):
                    scale_factor = 1.0
                else:
                    scale_factor = torch.rand([1], device=real_hr.device, dtype=real_hr.dtype) * \
                        (self.opt['scale_factor_end'] - self.opt['scale_factor_start']) + \
                        self.opt['scale_factor_start']

                real_lr = F.interpolate(real_hr, scale_factor=(1/scale_factor),
                    mode = "bilinear" if self.opt['mode'] == "2D" else "trilinear",
                    align_corners=False, recompute_scale_factor=False)
                lr_im = torch.from_numpy(np.transpose(to_img(real_lr, self.opt['mode']), 
                        [2, 0, 1])[0:3]).unsqueeze(0)
                lr_im = F.interpolate(lr_im, mode='nearest', size=hr_im.shape[2:])

                hr_coords, real_hr = to_pixel_samples(real_hr, flatten=False)
                cell_sizes = torch.ones_like(hr_coords)

                for i in range(cell_sizes.shape[-1]):
                    cell_sizes[:,:,i] *= 2 / real_shape[2+i]
                
                lr_upscaled = model(real_lr, hr_coords, cell_sizes)
                if(self.opt['mode'] == "2D"):
                    lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
                else:                    
                    lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)

                if(self.opt['residual_weighing']):
                    lr_interp = F.grid_sample(real_lr, hr_coords.flip(-1).unsqueeze(0), 
                        mode = "bilinear" if self.opt['mode'] == "2D" else "trilinear",
                        align_corners=False)
                    
                    rel_coord = make_residual_weight_grid(real_lr, hr_coords, self.opt['mode'])
                    #print(rel_coord)
                    lr_interp *= (1-rel_coord)
                    lr_upscaled *= rel_coord
                    lr_upscaled += lr_interp.detach()
                    
                
                sr_im = torch.from_numpy(np.transpose(to_img(lr_upscaled, 
                        self.opt['mode']),[2, 0, 1])[0:3]).unsqueeze(0)
                

                L1 = L1loss(torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0), real_hr)
                L1.backward()
                model_optim.step()
                optim_scheduler.step()
                psnr = PSNR(torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0), real_hr)
                
                if(step % self.opt['save_every'] == 0):
                    print("Epoch %i batch %i, sf: x%0.02f, L1: %0.04f, PSNR (dB): %0.02f" % \
                        (epoch, batch_num, scale_factor, L1.item(), psnr.item()))
                    writer.add_scalar('L1', L1.item(), step)
                    writer.add_images("LR, SR, HR", torch.cat([lr_im, sr_im, hr_im]), global_step=step)
                step += 1
            
            if(epoch % self.opt['save_every'] == 0):
                save_model(model, self.opt)
                print("Saved model")

        end_time = time.time()
        total_time = start_time - end_time
        print("Time to train: " + str(total_time))
        save_model(model, self.opt)
        print("Saved model")

    def train(self, model, dataset):
        torch.manual_seed(0b10101010101010101010101010101010)
        if(self.opt['train_distributed']):
            print("Training distributed across " + str(self.opt['gpus_per_node']) + " GPUs")
            os.environ['MASTER_ADDR'] = '127.0.0.1'              
            os.environ['MASTER_PORT'] = '29500' 
            mp.spawn(self.train_distributed,
                args=(model,self.opt,dataset),
                nprocs=self.opt['gpus_per_node'],
                join=True)
        else:
            print("Training on " + self.opt['device'])
            self.train_single(model, dataset)
         