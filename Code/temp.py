from torch.autograd.grad_mode import F
from utility_functions import str2bool
from options import *
from datasets import LocalDataset, LocalTemporalDataset
from models import GenericModel, load_model
from train import TemporalTrainer, Trainer, save_model
import argparse
import os
import h5py
import imageio
import numpy as np
from models import PositionalEncoding
import torch
from netCDF4 import Dataset

if __name__ == '__main__':
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    '''
    opt = Options.get_default()
    model = GenericModel(opt)

    m1_opt = load_options(os.path.join(save_folder, "isomag2D_RDN5_64_Shuffle1"))
    m2_opt = load_options(os.path.join(save_folder, "isomag2D_RDB5_64_LIIFskip"))

    m1 = load_model(m1_opt,"cuda:0")
    m2 = load_model(m2_opt,"cuda:0")
   
    model.feature_extractor = m1.feature_extractor
    model.upscaling_model = m2.upscaling_model
    
    opt['upscale_model'] = "LIIF_skip"
    opt['save_name'] = "RDN5_LIIF_finetune1thru4"

    save_model(model, opt)
    '''

    '''
    a= imageio.imread("pluto.png")
    print(a.shape)
    a = a.swapaxes(1,2).swapaxes(0,1)
    print(a.shape)
    a = a.astype(np.float32)[:,::2,::2]
    a /= 255.0
    f = h5py.File("pluto.h5", 'w')
    f['data'] = a
    f.close()
    '''
    
    '''
    opt = Options.get_default()
    pe = PositionalEncoding(opt)
    a = pe(torch.tensor([-0.5, -0.5]).unsqueeze(0).to("cuda"))
    b = pe(torch.tensor([-0.5, 0.5]).unsqueeze(0).to("cuda"))
    c = pe(torch.tensor([0.5, -0.5]).unsqueeze(0).to("cuda"))
    d = pe(torch.tensor([0.5, 0.5]).unsqueeze(0).to("cuda"))
    print(b - a)
    print(c - a)
    print(d - a)
    '''
