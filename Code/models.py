import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.utility_functions import VoxelShuffle, create_batchnorm_layer, create_conv_layer, weights_init
import os
from Code.options import save_options, load_options, Options

file_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(file_folder_path, "..")

input_folder = os.path.join(project_folder_path, "InputData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


def save_model(model, opt):
    folder_to_save_in = os.path.join(save_folder, opt['save_name'])
    if(not os.path.exists(folder_to_save_in)):
        os.makedirs(folder_to_save_in)
    print("Saving model to %s" % (folder_to_save_in))

    
    model_state = model.state_dict()
    torch.save(model_state, os.path.join(folder_to_save_in, "model.ckpt"))

    save_options(opt, folder_to_save_in)

def load_model(opt, device):
    model = None
    folder_to_load_from = os.path.join(save_folder, opt['save_name'])

    if not os.path.exists(folder_to_load_from):
        print("%s doesn't exist, load failed" % folder_to_load_from)
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(folder_to_load_from, "model.ckpt")):
        model_params = torch.load(os.path.join(folder_to_load_from, "model.ckpt"),
            map_location=device)

        model = MSR_Generator(opt)
        model.load_state_dict(model_params)

        print("Successfully loaded model")
    else:
        print("Warning: model.ckpt doesn't exists - can't load these model parameters")
   
    return model

class DenseBlock(nn.Module):
    def __init__(self, kernels, growth_channel, opt):
        super(DenseBlock, self).__init__()
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
        self.c1 = conv_layer(kernels, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c2 = conv_layer(kernels+growth_channel*1, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c3 = conv_layer(kernels+growth_channel*2, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c4 = conv_layer(kernels+growth_channel*3, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.lrelu = nn.LeakyReLU(0.2,inplace=True)
        self.final_conv = conv_layer(kernels+growth_channel*4, kernels, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])

    def forward(self,x):       
        c1_out = self.lrelu(self.c1(x))
        c2_out = self.lrelu(self.c2(torch.cat([x, c1_out], 1)))
        c3_out = self.lrelu(self.c3(torch.cat([x, c1_out, c2_out], 1)))
        c4_out = self.lrelu(self.c4(torch.cat([x, c1_out, c2_out, c3_out], 1)))
        final_out = self.final_conv(torch.cat([x, c1_out, c2_out, c3_out, c4_out], 1))
        return final_out

class RRDB(nn.Module):
    def __init__ (self,opt):
        super(RRDB, self).__init__()
        self.db1 = DenseBlock(opt['base_num_kernels'], int(opt['base_num_kernels']/4), opt)
        self.db2 = DenseBlock(opt['base_num_kernels'], int(opt['base_num_kernels']/4), opt)
        self.db3 = DenseBlock(opt['base_num_kernels'], int(opt['base_num_kernels']/4), opt)       
        self.B = torch.tensor([opt['B']])
        self.register_buffer('B_const', self.B)

    def forward(self,x):
        db1_out = self.db1(x) * self.B_const + x
        db2_out = self.db2(db1_out) * self.B_const + db1_out
        db3_out = self.db3(db2_out) * self.B_const + db2_out
        out = db3_out * self.B_const + x
        return out

class ESRGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(ESRGAN_Generator, self).__init__()
        self.opt = opt

        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
            self.pix_shuffle = nn.PixelShuffle(2)
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d

        self.c1 = conv_layer(opt['num_channels'], opt['base_num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
        self.blocks = []
        for i in range(opt['num_blocks']):
            self.blocks.append(RRDB(opt))
        self.blocks =  nn.ModuleList(self.blocks)
        
        self.c2 = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        # Upscaling happens between 2 and 3
        if(self.opt['mode'] == "2D"):
            fact = 4
        elif(self.opt['mode'] == "3D"):
            fact = 8
        if(self.opt['upsample_mode'] == "shuffle"):
            self.c2_vs = conv_layer(opt['base_num_kernels'], opt['base_num_kernels']*fact,
            stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
       
        self.c3 = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        self.final_conv = conv_layer(opt['base_num_kernels'], opt['num_channels'],
        stride=opt['stride'],padding=2,kernel_size=5)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x : torch.Tensor):
        x = self.c1(x)
        out = x.clone()
        for i, mod in enumerate(self.blocks):
            out = mod(out)
            
        out = self.c2(out)
        out = x + out

        if(self.opt['upsample_mode'] != "shuffle"):
            out = F.interpolate(out, scale_factor=2.0, 
            mode=self.opt['upsample_mode'], align_corners=True)
        elif(self.opt['upsample_mode'] == "shuffle"):
            out = self.c2_vs(out)
            if(self.opt['mode'] == "3D"):
                out = VoxelShuffle(out)
            elif(self.opt['mode'] == "2D"):
                out = self.pix_shuffle(out)
        
        out = self.lrelu(self.c3(out))
        out = self.final_conv(out)
        return out

class ESRGAN(nn.Module):
    def __init__(self, opt):
        super(ESRGAN, self).__init__()
        self.opt = opt
        self.G = ESRGAN_Generator(self.opt)
        self.D = SinGAN_Discriminator(self.opt)

    def forward(self, x : torch.Tensor):
        out = self.G(x)
        return out

class SinGAN_Discriminator(nn.Module):
    def __init__ (self, opt):
        super(SinGAN_Discriminator, self).__init__()

        use_sn = opt['regularization'] == "SN"
        if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d

        modules = []
        for i in range(opt['num_discrim_blocks']):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(
                    nn.Sequential(
                        create_conv_layer(conv_layer, 
                            opt['num_channels'], 
                            opt['num_kernels'], 
                            opt['kernel_size'], 
                            opt['stride'], 0, use_sn),
                        create_batchnorm_layer(batchnorm_layer, 
                            opt['num_kernels'], use_sn),
                            nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == opt['num_discrim_blocks']-1:  
                tail = nn.Sequential(
                    create_conv_layer(conv_layer, opt['num_kernels'], 1, 
                        opt['kernel_size'], opt['stride'], 0, use_sn)
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(
                    nn.Sequential(
                        create_conv_layer(conv_layer, 
                            opt['num_kernels'], 
                            opt['num_kernels'], 
                            opt['kernel_size'],
                            opt['stride'], 0, use_sn),
                        create_batchnorm_layer(batchnorm_layer, 
                            opt['num_kernels'], use_sn),
                            nn.LeakyReLU(0.2, inplace=True)
                    )
                )
        self.model =  nn.Sequential(*modules)

    def receptive_field(self):
        return (self.opt['kernel_size']-1)*self.opt['num_blocks']

    def forward(self, x):
        return self.model(x)

class MSR_Generator(nn.Module):
    def __init__(self, opt):
        super(MSR_Generator, self).__init__()
        self.opt = opt

        fc_in = 3
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
            fc_in = 4

        self.c1 = conv_layer(opt['num_channels'], opt['base_num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
        self.blocks = []
        for i in range(opt['num_blocks']):
            self.blocks.append(RRDB(opt))
        self.blocks =  nn.ModuleList(self.blocks)
        self.c2 = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'],
            stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.fc1 = nn.Linear(fc_in, 256)
        self.fc2 = nn.Linear(256, opt['base_num_kernels'] * opt['num_channels'] * opt['kernel_size'] * opt['kernel_size'])
        self.apply(weights_init)

    def forward(self, x, scale_factor):
        x = self.c1(x)
        features = x.clone()
        for i, mod in enumerate(self.blocks):
            features = mod(features)
            
        features = self.c2(features)
        features = x + features

        out_shape = x.shape
        for i in range(2, len(x.shape)):
            out_shape[i] = int(out_shape[i]*scale_factor)
        out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)

        sf_inds = torch.tensor(scale_factor, dtype=x.dtype, device=x.device).view(1, 1)  
        x_inds = torch.arange(0, out_shape[2], dtype=x.dtype, device=x.device).view(out_shape[2], 1)  
        y_inds = torch.arange(0, out_shape[3], dtype=x.dtype, device=x.device).view(out_shape[3], 1)  
        if(self.opt['mode'] == "2D"):
            x_inds = x_inds.repeat(out_shape[3], 1)
            y_inds = y_inds.repeat(out_shape[2], 1)
            sf_inds = sf_inds.repeat(out_shape[2] * out_shape[3], 1)
            weights_input = torch.cat([x_inds, y_inds, sf_inds], dim=1)
        if(self.opt['mode'] == "3D"):
            z_inds = torch.arange(0, out_shape[4], dtype=x.dtype, device=x.device)
            x_inds = x_inds.repeat(out_shape[3]*out_shape[4], 1)
            y_inds = y_inds.repeat(out_shape[2]*out_shape[4], 1)
            z_inds = z_inds.repeat(out_shape[2]*out_shape[3], 1)
            sf_inds = sf_inds.repeat(out_shape[2] * out_shape[3] * out_shape[4], 1)
            weights_input = torch.cat([x_inds, y_inds, z_inds, sf_inds], dim=1)

        weights = self.fc1(weights_input)
        weights = self.lrelu(weights)
        weights = self.fc2(weights)
        if(self.opt['mode'] == "2D"):
            weights = torch.reshape(weights, [x.shape[0], out_shape[1], out_shape[2], out_shape[3], self.opt['base_num_kernels'], 
                self.opt['kernel_size'], self.opt['kernel_size']])
        elif(self.opt['mode'] == "3D"):
            weights = torch.reshape(weights, [x.shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4],
                self.opt['base_num_kernels'], self.opt['kernel_size'], self.opt['kernel_size']])
        
        half_k_size = int(self.opt['kernel_size']/2)
        for i in range(out.shape[2]):
            i_prime = int(i/scale_factor)
            for j in range(out.shape[3]):
                j_prime = int(j/scale_factor)
                if(self.opt['mode'] == "2D"):                    
                    for c in range(out.shape[1]):
                        out[:,c,i,j] = weights[:,c,i,j,:,:,:] * \
                            features[:,:,i_prime-half_k_size:i_prime+half_k_size+1,j_prime-half_k_size:j_prime+half_k_size+1]
                elif(self.opt['mode'] == "3D"):
                    for k in range(out.shape[4]):                        
                        k_prime = int(k/scale_factor)
                        for c in range(out.shape[1]):
                            out[:,c,i,j,k] = weights[:,c,i,j,k,:,:,:] * \
                                features[:,:,i_prime-half_k_size:i_prime+half_k_size+1,
                                j_prime-half_k_size:j_prime+half_k_size+1,
                                k_prime-half_k_size:k_prime+half_k_size+1]

        return out


            