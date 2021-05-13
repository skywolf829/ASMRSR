from utility_functions import str2bool
from options import *
from datasets import LocalDataset, LocalTemporalDataset
from models import GenericModel, load_model
from train import TemporalTrainer, Trainer, save_model
import argparse
import os

if __name__ == '__main__':
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    opt = Options.get_default()
    model = GenericModel(opt)

    m1_opt = load_options(os.path.join(save_folder, "isomag2D_RDB5_64_Shuffle1"))
    m2_opt = load_options(os.path.join(save_folder, "isomag2D_RDB5_64_LIIFskip"))

    m1 = load_model(m1_opt,"cuda:0")
    m2 = loda_model(m2_opt,"cuda:0")
`   
    model.feature_extractor = m1.feature_extractor
    model.upscaling_model = m2.upscale_model
    
    opt['upscale_model'] = "LIIF"
    opt['save_name'] = "RDB5_LIIF_finetune"

    save_model(model, opt)
    `    