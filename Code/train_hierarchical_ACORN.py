from utility_functions import PSNR, str2bool, ssim
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
from octree import OctreeNodeList
from options import *
from datasets import LocalImplicitDataset
from models import HierarchicalACORN, PositionalEncoding
import argparse
from pytorch_memlab import LineProfiler, MemReporter, profile

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        torch.manual_seed(0b10101010101010101010101010101010)

    def train(self, model, dataset):
        
        print("Training on " + self.opt['device'])

        model = model.to(self.opt['device'])

        model_optim = optim.Adam(model.models[-1].parameters(), lr=self.opt["g_lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))

        #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
        #    milestones=[self.opt['epochs']/5, 
        #    2*self.opt['epochs']/5, 
        #    3*self.opt['epochs']/5, 
        #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])

        writer = SummaryWriter(os.path.join('tensorboard_ACORN',self.opt['save_name']))
        
        start_time = time.time()

        loss = nn.MSELoss().to(self.opt["device"])
        step = 0
        item = dataset[0].unsqueeze(0).to(self.opt['device'])
        print(item.shape)
        octree = OctreeNodeList(item)

        num_models = 1
        octree_subdiv_start = 4
        for _ in range(octree_subdiv_start):
            octree.next_depth_level()

        for model_num in range(num_models):
            for epoch in range(self.opt['epoch_number'], self.opt['epochs']):
                self.opt["epoch_number"] = epoch            
                model.zero_grad()            

                reconstructed = model(octree)
                l = loss(reconstructed, item)
                l.backward()
                model_optim.step()
                #optim_scheduler.step()
                
                if(step % 50 == 0):
                    with torch.no_grad():                        
                        psnr = PSNR(reconstructed, item, torch.tensor(1.0))
                        s = ssim(reconstructed, item)
                        print("Iteration %i, MSE: %0.04f, PSNR (dB): %0.02f, SSIM: %0.02f" % \
                            (epoch, l.item(), psnr.item(), s.item()))
                        writer.add_scalar('MSE', l.item(), step)
                        writer.add_scalar('PSNR', psnr.item(), step)
                        writer.add_scalar('SSIM', s.item(), step)             
                        if(len(model.models) > 1):
                            writer.add_image("Network"+str(len(model.models)-1)+"residual", 
                                (reconstructed-model.residual)[0].clamp_(0, 1), step)
                        writer.add_image("reconstruction", reconstructed[0].clamp_(0, 1), step)
                step += 1
            
                if(epoch % self.opt['save_every'] == 0):
                    save_model(model, self.opt)
                    print("Saved model")

            if(model_num < num_models-1):
                print("Adding higher-resolution model")
                model.errors.append(l.detach().item())
                model.add_model(opt)
                model.to(opt['device'])
                octree.next_depth_level()
                model_optim = optim.Adam(model.models[-1].parameters(), lr=self.opt["g_lr"], 
                    betas=(self.opt["beta_1"],self.opt["beta_2"]))
                #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
                #    milestones=[self.opt['epochs']/5, 
                #    2*self.opt['epochs']/5, 
                #    3*self.opt['epochs']/5, 
                #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])
                for param in model.models[-2].parameters():
                    param.requires_grad = False
                self.opt['epoch_number'] = 0
                model.residual = None
       

        end_time = time.time()
        total_time = end_time - start_time
        print("Time to train: %0.01f minutes" % (total_time/60))
        save_model(model, self.opt)
        print("Saved model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,help='The type of input - 2D, 3D')
    parser.add_argument('--feat_model',default=None,type=str,help='Feature extraction model')
    parser.add_argument('--upscale_model',default=None,type=str,help='Upscaling model')
    parser.add_argument('--residual_weighing',default=None,type=str2bool,help='Use residual weighing')
    parser.add_argument('--data_folder',default=None,type=str,help='File to train on')
    parser.add_argument('--num_training_examples',default=None,type=int,help='Frames to use from training file')
    parser.add_argument('--save_folder',default=None, help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None, help='The name for the folder to save the model')
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels to use')
    parser.add_argument('--spatial_downscale_ratio',default=None,type=float,help='Ratio for spatial downscaling')
    parser.add_argument('--min_dimension_size',default=None,type=int,help='Minimum dimension size')
    parser.add_argument('--scaling_mode',default=None,type=str,help='Scaling mode, learned, magnitude, channel, or none')
    parser.add_argument('--fine_tuning',default=None,type=str2bool,help='Only train upscaling model')

    parser.add_argument('--num_lstm_layers',default=None,type=int,help='num lstm layers')
    parser.add_argument('--training_seq_length',default=None,type=int,help='length of sequence to train LSTM with')

    parser.add_argument('--num_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--base_num_kernels',default=None,type=int, help='Num conv kernels in lowest layer')
    parser.add_argument('--pre_padding',default=None,type=str2bool, help='Padding before entering network')
    parser.add_argument('--kernel_size',default=None, type=int,help='Conv kernel size')
    parser.add_argument('--stride',default=None, type=int,help='Conv stride length')
    
    
    parser.add_argument('--scale_factor_start',default=None, type=float,help='Where to start in-dist training')
    parser.add_argument('--scale_factor_end',default=None, type=float,help='Where to stop in-dist training') 
    parser.add_argument('--cropping_resolution',default=None, type=int,help='Cropping resolution') 
    parser.add_argument('--time_cropping_resolution',default=None, type=int, help="Crop time")

    parser.add_argument('--x_resolution',default=None, type=int,help='x')    
    parser.add_argument('--y_resolution',default=None, type=int,help='y')
    parser.add_argument('--z_resolution',default=None, type=int,help='z')
    
    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--device',type=str,default=None, help='Device to use')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--ranking',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--save_generators',default=None, type=str2bool,help='Whether or not to save generators')
    parser.add_argument('--save_discriminators',default=None, type=str2bool,help='Whether or not to save discriminators')

    parser.add_argument('--alpha_1',default=None, type=float,help='Reconstruction loss coefficient')
    parser.add_argument('--alpha_2',default=None, type=float,help='Adversarial loss coefficient')
    parser.add_argument('--alpha_3',default=None, type=float,help='Soft physical constraint loss coefficient')
    parser.add_argument('--alpha_4',default=None, type=float,help='mag and angle loss coeff')
    parser.add_argument('--alpha_5',default=None, type=float,help='gradient loss coeff')
    parser.add_argument('--alpha_6',default=None, type=float,help='streamline loss coeff')
    parser.add_argument('--streamline_res',default=None, type=int,help='num seeds per dim')
    parser.add_argument('--streamline_length',default=None, type=int,help='timesteps to do streamlines')
    parser.add_argument('--adaptive_streamlines',default=None, type=str2bool,help='Adaptive particle sampling for streamlines')
    parser.add_argument('--periodic',default=None, type=str2bool,help='is data periodic')
    parser.add_argument('--generator_steps',default=None, type=int,help='Number of generator steps to take')
    parser.add_argument('--discriminator_steps',default=None, type=int,help='Number of discriminator steps to take')
    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--minibatch',default=None, type=int,help='Size of minibatch to train on')
    parser.add_argument('--num_workers',default=None, type=int,help='Number of workers for dataset loader')
    parser.add_argument('--g_lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--d_lr',default=None, type=float,help='Learning rate for the discriminator')
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')
    parser.add_argument('--gamma',default=None, type=float,help='')
    parser.add_argument('--regularization',default=None, type=str,help='')
    parser.add_argument('--physical_constraints',default=None,type=str,help='none, soft, or hard')
    parser.add_argument('--patch_size',default=None, type=int,help='Patch size for inference')
    parser.add_argument('--training_patch_size',default=None, type=int,help='Patch size for training')
    parser.add_argument('--upsample_mode',default=None, type=str,help='Method for upsampling')
    parser.add_argument('--zero_noise',default=None, type=str2bool,help='Whether or not to use zero noise during upscaling')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')

    args = vars(parser.parse_args())

    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    prof = LineProfiler()
    prof.enable()

    if(args['load_from'] is None):
        opt = Options.get_default()
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        dataset = LocalImplicitDataset(opt)
        model = HierarchicalACORN(opt)

    trainer = Trainer(opt)
    trainer.train(model, dataset)

    print(prof.display())
    prof.disable()