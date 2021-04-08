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
        model = DDP(model, device_ids=[rank]) 

        model_optim = optim.Adam(model.parameters(), lr=self.opt["g_lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
            milestones=[0.8*self.opt['epochs']-self.opt['epoch_number']],gamma=self.opt['gamma'])

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

        for epoch in range(opt['epoch_number'], opt['epochs']):
            opt["epoch_number"] = epoch
            for batch_num, real_hr in enumerate(dataloader):
                model.zero_grad()
                real_hr = real_hr.to(opt['device'])
                scale_factor = torch.rand([1]) * \
                    (self.opt['scale_factor_end'] - self.opt['scale_factor_start']) + \
                        opt['scale_factor_start']
                real_lr = F.interpolate(real_hr, scale_factor=(1/scale_factor),
                    mode = "bilinear" if opt['mode'] == "2D" else "trilinear")

                lr_upscaled = model(real_lr, scale_factor)

                L1 = L1loss(lr_upscaled, real_hr)
                L1.backward()
                model_optim.step()
                optim_scheduler.step()

                if(batch_num % 5 == 0 and rank == 0):
                    print("L1: " + L1.item())
                    writer.add_scalar('L1', L1.item())
            if(rank == 0):
                save_model(model, opt)
                print("Saved model")

        end_time = time.time()
        total_time = start_time - end_time
        if(rank == 0):
            print("Time to train: " + str(total_time))

    def train_single(self, model, dataset):
        model = model.to(self.opt['device'])

        model_optim = optim.Adam(model.parameters(), lr=self.opt["g_lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
            milestones=[0.8*self.opt['epochs']-self.opt['epoch_number']],gamma=self.opt['gamma'])

        writer = SummaryWriter(os.path.join('tensorboard',self.opt['save_name']))

        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=True,
            num_workers=self.opt["num_workers"],
            pin_memory=True
        )
        L1loss = nn.L1Loss().to(self.opt["device"])

        for epoch in range(self.opt['epoch_number'], self.opt['epochs']):
            self.opt["epoch_number"] = epoch
            for batch_num, real_hr in enumerate(dataloader):
                model.zero_grad()
                real_hr = real_hr.to(self.opt['device'])
                print("Full shape : " + str(real_hr.shape))
                scale_factor = torch.rand([1]) * \
                    (self.opt['scale_factor_end'] - self.opt['scale_factor_start']) + \
                        self.opt['scale_factor_start']
                print("Scale factor: " + str(scale_factor))
                real_lr = F.interpolate(real_hr, scale_factor=(1/scale_factor),
                    mode = "bilinear" if self.opt['mode'] == "2D" else "trilinear",
                    align_corners=True, recompute_scale_factor=True)
                print("LR shape : " + str(real_lr.shape))

                lr_upscaled = model(real_lr, scale_factor)

                L1 = L1loss(lr_upscaled, real_hr)
                L1.backward()
                model_optim.step()
                optim_scheduler.step()

                if(batch_num % 5 == 0):
                    print("L1: " + L1.item())
                    writer.add_scalar('L1', L1.item())
            
            save_model(model, self.opt)
            print("Saved model")

        end_time = time.time()
        total_time = start_time - end_time
        print("Time to train: " + str(total_time))

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
         