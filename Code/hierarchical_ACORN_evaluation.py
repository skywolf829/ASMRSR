from math import log10
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
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

if __name__ == '__main__':
    
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    load_from = "temp"

    opt = load_options(os.path.join(save_folder, load_from))
    opt["device"] = "cuda:0"
    opt["save_name"] = load_from

    dataset = LocalImplicitDataset(opt)
    model = HierarchicalACORN(opt)
    model_params = torch.load(os.path.join(os.path.join(save_folder, load_from), "model.ckpt"),
        map_location="cuda:0")
    item = dataset[0].unsqueeze(0).to(opt['device'])
    octree = OctreeNodeList(item)
    octree = octree.load(os.path.join(save_folder, load_from))
    octree.data = item
    model.load_state_dict(model_params)

    print("Everything loaded")

    img = model.get_full_img(octree)
