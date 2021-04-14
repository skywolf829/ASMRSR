import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["mode"]                    = "2D"      # What SinGAN to use - 2D or 3D
        opt["data_folder"]             = "TrainingData/isomag2D"
        opt['scaling_mode']            = None # magnitude, channel, learned, none
        opt['load_data_at_start']      = False
        opt['single_shot']            =  False
        opt["save_folder"]             = "SavedModels"
        opt["save_name"]               = "Temp"    # Folder that the model will be saved to
        opt["num_channels"]            = 1
        opt["spatial_downscale_ratio"] = 0.5       # Spatial downscale ratio between levels
        opt["min_dimension_size"]      = 16        # Smallest a dimension can go to upscale from
        opt["cropping_resolution"]     = 48
        opt["train_date_time"]         = None      # The day/time the model was trained (finish time)
        
        opt['dataset_name']            = "isotropic1024coarse"
        opt['num_dataset_timesteps']   = 100
        opt['x_resolution']            = 1024
        opt['y_resolution']            = 1024
        opt['z_resolution']            = 1
        opt['ts_skip']                 = 10
        opt['num_dims']                = 3
        opt['random_flipping']         = True
        opt['num_networked_workers']   = 4

        opt["num_workers"]             = 2

        # generator info
        opt["num_blocks"]              = 10
        opt['num_discrim_blocks']      = 5
        opt["base_num_kernels"]        = 64        # Num of kernels in smallest scale conv layers
        opt["pre_padding"]             = False         # Padding on conv layers in the GAN
        opt["kernel_size"]             = 3
        opt["padding"]                 = 1
        opt["stride"]                  = 1
        opt['conv_groups']             = 1
        opt['separate_chans']          = False
        opt['B']                      = 0.2

        opt['num_lstm_layers']         = 3
        opt['training_seq_length']     = 3
        opt['temporal_direction']     = "forward"
        opt['temporal_generator']     = "TSRTVD"


        opt["n"]                       = 0         # Number of scales in the heirarchy, defined by the input and min_dimension_size
        opt["resolutions"]             = []        # The scales for the GAN
        opt["downsample_mode"]         = "average_pooling"
        opt["upsample_mode"]           = "trilinear"

        opt['scale_factor_start']      = 1.1
        opt['scale_factor_end']        = 8.0

        opt["train_distributed"]       = False
        opt["device"]                  = "cuda:0"
        opt["gpus_per_node"]           = 8
        opt["num_nodes"]               = 1
        opt["ranking"]                 = 0

        opt["save_generators"]         = True
        opt["save_discriminators"]     = True
        opt["physical_constraints"]    = "none"
        opt["patch_size"]              = 128
        opt["training_patch_size"]     = 96
        opt["regularization"]          = "GP" #Either TV (total variation) or GP (gradient penalty) or SN 

        # GAN training info
        opt["alpha_1"]                 = 1       # Reconstruction loss coefficient
        opt["alpha_2"]                 = 0.1        # Adversarial loss coefficient
        opt["alpha_3"]                 = 0        # Soft physical loss coefficient
        opt["alpha_4"]                 = 0        # mag_and_angle loss
        opt["alpha_5"]                 = 0          # first derivative loss coeff
        opt["alpha_6"]                 = 0         # Lagrangian transport loss

        opt["adaptive_streamlines"]    = False
        opt['streamline_res']          = 100
        opt['streamline_length']       = 5

        opt['periodic']                = False
        opt["generator_steps"]         = 1
        opt["discriminator_steps"]     = 1
        opt["epochs"]                  = 50
        opt["minibatch"]               = 1        # Minibatch for training
        opt["g_lr"]                    = 0.0001    # Learning rate for GAN generator
        opt["d_lr"]                    = 0.0004    # Learning rate for GAN discriminator
        opt["beta_1"]                  = 0.5
        opt["beta_2"]                  = 0.999
        opt["gamma"]                   = 0.1

        # Info during training (to continue if it stopped)
        opt["scale_in_training"]       = 0
        opt["iteration_number"]        = 0
        opt["epoch_number"]            = 0
        opt["save_every"]              = 100
        opt["save_training_loss"]      = True

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
