import torch
import torch.nn as nn
import torch.nn.functional as F
from utility_functions import VoxelShuffle, create_batchnorm_layer, create_conv_layer, weights_init, make_coord
import os
from options import save_options, load_options, Options
import numpy as np
from math import ceil
file_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(file_folder_path, "..")

input_folder = os.path.join(project_folder_path, "TrainingData")
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
        keys = list(model_params.keys())
        for k in keys:
            if "module." in k:
                model_params[k[7:]] = model_params[k]
                del model_params[k]
        model = LIIF_Generator(opt)
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

class RDB(nn.Module):
    def __init__ (self,in_c,out_c,opt):
        super(RDB, self).__init__()
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d

        self.block = nn.Sequential(
            conv_layer(in_c, out_c, kernel_size=opt['kernel_size'], padding=1),
            nn.ReLU(),
            conv_layer(out_c, out_c, kernel_size=opt['kernel_size'], padding=1)
        )

    def forward(self, x):
        out = self.block(x)
        return out + x

class RRDN(nn.Module):
    def __init__ (self,opt):
        super(RRDN, self).__init__()
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d

        self.first_conv = conv_layer(opt['num_channels'], opt['base_num_kernels'],
            stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        self.blocks = nn.ModuleList()
        for i in range(opt['num_blocks']):
            self.blocks.append(RRDB(opt))
        self.blocks = nn.Sequential(*self.blocks)
        self.final_conv = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'],
            stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.first_conv(x)
        out = self.blocks(x)
        out = self.final_conv(out)
        return out + x

class RDN(nn.Module):
    def __init__ (self,opt):
        super(RDN, self).__init__()
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d

        self.first_conv = conv_layer(opt['num_channels'], opt['base_num_kernels'], kernel_size=opt['kernel_size'], padding=1)
        self.blocks = nn.ModuleList()
        for i in range(opt['num_blocks']):
            self.blocks.append(RDB(opt['base_num_kernels'], opt['base_num_kernels'], opt))
        self.blocks = nn.Sequential(*self.blocks)
        self.final_conv = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'], kernel_size=opt['kernel_size'], padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        out = self.blocks(x)
        out = self.final_conv(out)
        return out+x

class LIIF_Generator(nn.Module):
    def __init__(self, opt):
        super(LIIF_Generator, self).__init__()
        self.opt = opt
        n_dims = 2

        self.feature_extractor = RDN(opt)     
        self.LIIF = nn.ModuleList([
            nn.Linear(opt['base_num_kernels']*(3**n_dims)+n_dims+n_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, opt["num_channels"])
        ])
        self.LIIF = nn.Sequential(*self.LIIF)
        self.apply(weights_init)

    def forward(self, lr, locations, cell_sizes):
        #lr_mean = lr.mean()
        features = self.feature_extractor(lr)
        #print("Features shape : " + str(features.shape))

        n_dims = len(features.shape[2:])
        if(self.opt['mode'] == "2D"):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            #print("Padded features " + str(features.shape))
            features = F.unfold(features, 3, padding=0).view(
                features.shape[0], features.shape[1] * (3**n_dims), features.shape[2]-2, features.shape[3]-2)
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0).view(
                features.shape[0], features.shape[1] * (3**n_dims), features.shape[2]-2, 
                features.shape[3]-2, features.shape[4]-2)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]
        #print("Features unfolded : " + str(features.shape))
        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
        #print("Feat coord before stuff" + str(feat_coord.shape))
        if(self.opt['mode'] == "2D"):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])
        #print("Feat coord " + str(feat_coord.shape))
        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if(self.opt['mode'] == "2D"):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0),
                        mode='nearest', align_corners=False)[0]
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0),
                        mode='nearest', align_corners=False)[0]
                    #print("Q coord: " + str(q_coord.shape))
                    rel_coord = locations - q_coord.permute(1, 2, 0)

                    rel_coord[:, :, 0] *= features.shape[2]
                    rel_coord[:, :, 1] *= features.shape[3]
                    
                    #print("Rel coords: " + str(rel_coord.shape))
                    fc_input = torch.cat([q_feat.permute(1, 2, 0).contiguous(), rel_coord], dim=-1)
                    #print("fc_input : " + str(fc_input.shape))

                    rel_cell = cell_sizes.clone()
                    rel_cell[:, :, 0] *= features.shape[2]
                    rel_cell[:, :, 1] *= features.shape[3]
                    #print("Rel_cell: " + str(rel_cell.shape))
                    fc_input = torch.cat([fc_input, rel_cell], dim=-1)

                    #print("fc_input : " + str(fc_input.shape))
                    pred = self.LIIF(fc_input)                    
                    preds.append(pred)
                    
                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])#.flatten()
                    areas.append(area + 1e-9)
                else:
                    for vz in vz_lst:
                        loc_ = locations.clone()
                        loc_[:, :, :, 0] += vx * rx + eps_shift
                        loc_[:, :, :, 1] += vy * ry + eps_shift
                        loc_[:, :, :, 2] += vz * rz + eps_shift
                        loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                        q_feat = F.grid_sample(
                            features, loc_.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :] \
                            .permute(0, 2, 1)
                        q_coord = F.grid_sample(
                            feat_coord, loc_.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :] \
                            .permute(0, 2, 1)
                        rel_coord = locations - q_coord
                        rel_coord[:, :, 0] *= features.shape[-2]
                        rel_coord[:, :, 1] *= features.shape[-1]
                        inp = torch.cat([q_feat, rel_coord], dim=-1)

                        if self.cell_decode:
                            rel_cell = cell_sizes.clone()
                            rel_cell[:, :, 0] *= features.shape[-2]
                            rel_cell[:, :, 1] *= features.shape[-1]
                            inp = torch.cat([inp, rel_cell], dim=-1)

                        bs, q = locations.shape[:2]
                        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                        preds.append(pred)

                        area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                        areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if(self.opt['mode'] == "2D"):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
        #print("Ret " + str(ret.shape))
        return ret#+lr_mean
