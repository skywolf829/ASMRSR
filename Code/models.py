from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.activation import LeakyReLU, SiLU
from utility_functions import  create_batchnorm_layer, create_conv_layer, weights_init, make_coord, \
    bilinear_interpolate, trilinear_interpolate
import os
from options import save_options
from einops.layers.torch import Rearrange
from math import pi
from pytorch_memlab import LineProfiler, MemReporter, profile, profile_every
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

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
    model = GenericModel(opt)
    folder_to_load_from = os.path.join(save_folder, opt['save_name'])

    if not os.path.exists(folder_to_load_from):
        print("%s doesn't exist, load failed" % folder_to_load_from)
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(folder_to_load_from, "model.ckpt")):
        model_params = torch.load(os.path.join(folder_to_load_from, "model.ckpt"),
            map_location=device)
        keys = list(model_params.keys())
        #print(keys)
        for k in keys:
            if "module." in k:
                model_params[k[7:]] = model_params[k]
                del model_params[k]
        '''
        keys = list(model_params.keys())
        for k in keys:
            if opt['upscale_model'] in k:
                model_params["upscaling_model."+k] = model_params[k]
                del model_params[k]
        '''

        model.load_state_dict(model_params)

        print("Successfully loaded model")
    else:
        print("Warning: model.ckpt doesn't exists - can't load these model parameters")
   
    return model

class ResidueBlock(nn.Module):
    def __init__(self, channels_in, channels_out, opt):
        super(ResidueBlock, self).__init__()
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        else:
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d
        self.block = nn.Sequential(
            conv_layer(channels_in, channels_out, 1),
            batchnorm_layer(channels_out),
            nn.ReLU(),
            conv_layer(channels_out, channels_out, 3, padding=1),
            batchnorm_layer(channels_out),
            nn.ReLU(),
            conv_layer(channels_out, channels_out, 1),
            batchnorm_layer(channels_out)
        )
        if(channels_in != channels_out):
            self.c1 = conv_layer(channels_in, channels_out, 3, padding=1)
        else:
            self.c1 = nn.Identity()

    def forward(self,x):       
        y = self.block(x)
        return F.relu(self.c1(x)+y)

class DenseBlock(nn.Module):
    def __init__(self, kernels, growth_channel, opt):
        super(DenseBlock, self).__init__()
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
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

        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
            self.pix_shuffle = nn.PixelShuffle(2)
        else:
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
        if("2D" in self.opt['mode']):
            fact = 4
        else:
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
            if("2D" in self.opt['mode']):
                out = self.pix_shuffle(out)            
            else:
                out = VoxelShuffle(out)
        
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
        if("2D" in opt['mode'] or opt['mode'] == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        else:
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
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
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
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
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
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
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

class RDN_skip(nn.Module):
    def __init__ (self,opt):
        super(RDN_skip, self).__init__()
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
            conv_layer = nn.Conv3d

        self.first_conv = conv_layer(opt['num_channels'], opt['base_num_kernels'], kernel_size=opt['kernel_size'], padding=1)
        self.blocks = nn.ModuleList()
        for i in range(opt['num_blocks']):
            self.blocks.append(RDB(opt['base_num_kernels'], opt['base_num_kernels'], opt))
        self.blocks = nn.Sequential(*self.blocks)
        self.final_conv = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'], kernel_size=opt['kernel_size'], padding=1)

    def forward(self, x):
        x1 = self.first_conv(x)
        out = self.blocks(x1)
        out = self.final_conv(out)
        out = out + x1
        out = torch.cat([out, x.detach()], dim=1)
        return out

class UNet(nn.Module):
    def __init__ (self,opt):
        super(UNet, self).__init__()
        self.opt = opt
        self.blocks = nn.ModuleList(
            [
                ResidueBlock(opt['num_channels'], 16, opt),
                ResidueBlock(16, 32, opt),
                ResidueBlock(32, 64, opt),
                ResidueBlock(64, 128, opt),

                ResidueBlock(128, 256, opt),

                ResidueBlock(256, 128, opt),
                ResidueBlock(256, 64, opt),
                ResidueBlock(128, 32, opt),
                ResidueBlock(64, 16, opt)
            ]
        )
        if("2D" in self.opt['mode']):
            self.maxpool = nn.MaxPool2d(2, stride=2)
        else:
            self.maxpool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x0 = self.blocks[0](x) # c=16
        x1 = self.maxpool(self.blocks[1](x0))  # c=32
        x2 = self.maxpool(self.blocks[2](x1))  # c=64
        x3 = self.maxpool(self.blocks[3](x2))  # c=128

        x4 = self.maxpool(self.blocks[4](x3))  # c=256

        y3 = self.blocks[5](F.interpolate(x4, size=x3.shape[2:], mode='nearest'))
        y3 = torch.cat([y3, x3], dim=1)

        y2 = self.blocks[6](F.interpolate(y3, size=x2.shape[2:], mode='nearest'))
        y2 = torch.cat([y2, x2], dim=1)

        y1 = self.blocks[7](F.interpolate(y2, size=x1.shape[2:], mode='nearest'))
        y1 = torch.cat([y1, x1], dim=1)

        y0 = self.blocks[8](F.interpolate(y1, size=x0.shape[2:], mode='nearest'))
        y0 = torch.cat([y0, x0], dim=1)

        return y0

class Identity(nn.Module):
    def __init__ (self,opt):
        super(Identity, self).__init__()
        self.opt = opt

    def forward(self, x):
        return x

class MFFN(nn.Module):
    def __init__(self, opt):
        super(MFFN, self).__init__()
        self.continuous = True
        self.opt = opt
        num_input = 32
        if("2D" in self.opt['mode']):
            num_input += 2
        else:
            num_input += 3
        self.MFFN = nn.ModuleList([
            nn.Linear(num_input, 512),
            nn.SiLU(),
            nn.Linear(num_input+512, 256),
            nn.SiLU(),
            nn.Linear(num_input+256, 128),
            nn.SiLU(),
            nn.Linear(num_input+128, 64),
            nn.SiLU(),
            nn.Linear(num_input+64, 32),
            nn.SiLU(),
            nn.Linear(32, opt['num_channels'])
        ])
        
        self.apply(weights_init)

    def forward(self, features, locations, cell_sizes=None):
        lr_shape = features.shape
        
        context_vectors = F.grid_sample(features, locations.flip(-1).unsqueeze(0), 
            mode='bilinear' if "2D" in self.opt['mode'] else 'trilinear',
            align_corners=False)
        context_vectors = torch.cat([context_vectors, 
            locations.flip(-1).permute(2, 0, 1).unsqueeze(0)], dim=1)
        if("2D" in self.opt['mode']):
            context_vectors = context_vectors.permute(0, 2, 3, 1).contiguous()
        else:
            context_vectors = context_vectors.permute(0, 2, 3, 4, 1).contiguous()
        x = self.MFFN[1](self.MFFN[0](context_vectors))
        x = self.MFFN[3](self.MFFN[2](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[5](self.MFFN[4](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[7](self.MFFN[6](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[9](self.MFFN[8](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[10](x)
        return x[0]

class LIIF(nn.Module):
    def __init__(self, opt):
        super(LIIF, self).__init__()
        self.continuous = True
        self.opt = opt
        n_dims = 2
        latent_vector_size = opt['base_num_kernels']*(3**n_dims)+n_dims+n_dims
        if("skip" in opt['feat_model']):
            latent_vector_size += 3**n_dims
        self.LIIF = nn.ModuleList([
            nn.Linear(latent_vector_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, opt["num_channels"])
        ])
        #self.LIIF = nn.Sequential(*self.LIIF)
        self.apply(weights_init)

    def LIIF_forward(self, fc_input):
        x = fc_input
        for i in range(len(self.LIIF)-1):
            x = self.LIIF[i](x)
            x = F.relu(x)
        x = self.LIIF[-1](x)

        return x

    def forward(self, features, locations, cell_sizes):
        lr_shape = features.shape
        
        if("2D" in self.opt['mode']):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]

        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
            
        if("2D" in self.opt['mode']):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if("2D" in self.opt['mode']):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]
                        
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
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
                    pred = self.LIIF_forward(fc_input)                    
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
        if("2D" in self.opt['mode']):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

class LIIF_skip_posonly(nn.Module):
    def __init__(self, opt):
        super(LIIF_skip_posonly, self).__init__()
        self.continuous = True
        self.opt = opt
        n_dims = 2
        latent_vector_size = opt['base_num_kernels']*(3**n_dims)+n_dims+n_dims
        if("skip" in opt['feat_model']):
            latent_vector_size += 3**n_dims
        self.pos_skip_size = n_dims+n_dims
        self.LIIF = nn.ModuleList([
            nn.Linear(latent_vector_size, 256),
            nn.Linear(256+self.pos_skip_size, 256),
            nn.Linear(256+self.pos_skip_size, 256),
            nn.Linear(256+self.pos_skip_size, 256),
            nn.Linear(256, opt["num_channels"])
        ])
        #self.LIIF = nn.Sequential(*self.LIIF)
        self.apply(weights_init)

    def LIIF_forward(self, fc_input):
        x = fc_input
        pos_info = x[-self.pos_skip_size:].clone().detach()
        for i in range(len(self.LIIF)-1):
            x = self.LIIF[i](x)
            x = F.relu(x)
            if(i < len(self.LIIF)-2):
                x = torch.cat([x, pos_info.clone().detach()], dim=-1)
        x = self.LIIF[-1](x)
        return x

    def forward(self, features, locations, cell_sizes):
        lr_shape = features.shape
        
        if("2D" in self.opt['mode']):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]

        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
            
        if("2D" in self.opt['mode']):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if("2D" in self.opt['mode']):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]
                        
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
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
                    pred = self.LIIF_forward(fc_input)                    
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
        if("2D" in self.opt['mode']):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

class LIIF_skip(nn.Module):
    def __init__(self, opt):
        super(LIIF_skip, self).__init__()
        self.continuous = True
        self.opt = opt
        n_dims = 2
        latent_vector_size = opt['base_num_kernels']*(3**n_dims)+n_dims+n_dims
        if("skip" in opt['feat_model']):
            latent_vector_size += 3**n_dims
        self.LIIF = nn.ModuleList([
            nn.Linear(latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256, opt["num_channels"])
        ])
        #self.LIIF = nn.Sequential(*self.LIIF)
        self.apply(weights_init)

    def LIIF_forward(self, fc_input):
        fc_input_orig = fc_input.clone()
        x = fc_input
        for i in range(len(self.LIIF)-1):
            x = self.LIIF[i](x)
            x = F.relu(x)
            if(i < len(self.LIIF)-2):
                x = torch.cat([x, fc_input_orig], dim=-1)
        x = self.LIIF[-1](x)
        return x

    def forward(self, features, locations, cell_sizes):
        lr_shape = features.shape
        
        if("2D" in self.opt['mode']):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]

        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
            
        if("2D" in self.opt['mode']):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if("2D" in self.opt['mode']):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]
                        
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
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
                    pred = self.LIIF_forward(fc_input)                    
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
        if("2D" in self.opt['mode']):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

class LIIF_skip_res(nn.Module):
    def __init__(self, opt):
        super(LIIF_skip_res, self).__init__()
        self.continuous = True
        self.opt = opt
        n_dims = 2
        latent_vector_size = opt['base_num_kernels']*(3**n_dims)+n_dims+n_dims
        if("skip" in opt['feat_model']):
            latent_vector_size += 3**n_dims
        self.LIIF = nn.ModuleList([
            nn.Linear(latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256+latent_vector_size, 256),
            nn.Linear(256, opt["num_channels"])
        ])
        #self.LIIF = nn.Sequential(*self.LIIF)
        self.apply(weights_init)

    def LIIF_forward(self, fc_input):
        fc_input_orig = fc_input.clone()
        x = fc_input
        for i in range(len(self.LIIF)-1):
            y = self.LIIF[i](x)
            y = F.relu(y)
            y = y + x[..., 0:256]
            if(i < len(self.LIIF)-2):
                x = torch.cat([y, fc_input_orig], dim=-1)
            else:
                x = y
        x = self.LIIF[-1](x)
        return x

    def forward(self, features, locations, cell_sizes):
        lr_shape = features.shape
        
        if("2D" in self.opt['mode']):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]

        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
            
        if("2D" in self.opt['mode']):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if("2D" in self.opt['mode']):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]
                        
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
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
                    pred = self.LIIF_forward(fc_input)                    
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
        if("2D" in self.opt['mode']):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

class UltraSR(nn.Module):
    def __init__(self, opt):
        super(UltraSR, self).__init__()
        self.continuous = True
        self.opt = opt
        n_dims = 2
        self.pos_skip_size = n_dims
        self.num_fourier_weights = 12
        self.fourier_weights = torch.arange(1, self.num_fourier_weights+1, 1, 
            dtype=torch.float32, device=opt['device'])
        self.fourier_weights.requires_grad=True
        latent_vector_size = opt['base_num_kernels']*(3**n_dims) + n_dims + n_dims*self.num_fourier_weights*2
        if("skip" in opt['feat_model']):
            latent_vector_size += 3**n_dims

        self.UltraSR = nn.ModuleList([
            nn.Linear(latent_vector_size, 256),
            nn.Linear(256+n_dims*self.num_fourier_weights*2, 256),
            nn.Linear(256+n_dims*self.num_fourier_weights*2, 256),
            nn.Linear(256+n_dims*self.num_fourier_weights*2, 256),
            nn.Linear(256, opt["num_channels"])
        ])

        self.apply(weights_init)

    def UltraSR_forward(self, feats, rel_coord, rel_cell):

        rel_coord = rel_coord.unsqueeze(-1)
        rel_coord = rel_coord.repeat([1]*(len(rel_coord.shape)-1) + [self.num_fourier_weights])

        sin_rel_coord = torch.sin(rel_coord*self.fourier_weights)
        cos_rel_coord = torch.cos(rel_coord*self.fourier_weights)


        x = torch.cat([feats, rel_cell,
            sin_rel_coord.flatten(-2), cos_rel_coord.flatten(-2)], 
            dim=-1)
        for i in range(len(self.UltraSR)-1):
            x = self.UltraSR[i](x)
            x = F.relu(x)
            if(i < len(self.UltraSR)-2):
                x = torch.cat([x, sin_rel_coord.flatten(-2), cos_rel_coord.flatten(-2)], dim=-1)
        x = self.UltraSR[-1](x)
        return x

    def forward(self, features, locations, cell_sizes):
        lr_shape = features.shape
        
        if("2D" in self.opt['mode']):
            features = F.pad(features, [1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])
        else:
            features = F.pad(features, [1, 1, 1, 1, 1, 1], mode='reflect')
            features = F.unfold(features, 3, padding=0)
            features = features.view(
                features.shape[0], features.shape[1], *lr_shape[2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = (2 / features.shape[2]) / 2
        ry = (2 / features.shape[3]) / 2
        if(self.opt['mode']== "3D"):
            rz = 2 / features.shape[4] / 2
            vz_lst = [-1, 1]

        feat_coord = make_coord(features.shape[2:], device=self.opt['device'],
            flatten=False)
            
        if("2D" in self.opt['mode']):
            feat_coord = feat_coord.permute(2, 0, 1).\
                unsqueeze(0).expand(features.shape[0], 2, *features.shape[2:])
        else:
            feat_coord = feat_coord.permute(3, 0, 1, 2).\
                unsqueeze(0).expand(features.shape[0], 3, *features.shape[2:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                if("2D" in self.opt['mode']):
                    loc_ = locations.clone()
                    #print("Loc: " + str(loc_.shape))
                    loc_[:, :, 0] += vx * rx + eps_shift
                    loc_[:, :, 1] += vy * ry + eps_shift
                    loc_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        features, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]
                        
                    #print("Q feat: " + str(q_feat.shape))
                    q_coord = F.grid_sample(
                        feat_coord, loc_.flip(-1).unsqueeze(0).repeat(lr_shape[0], 
                        *[1]*(len(lr_shape)-1)),
                        mode='nearest', align_corners=False)[0]

                    #print("Q coord: " + str(q_coord.shape))
                    rel_coord = locations - q_coord.permute(1, 2, 0)

                    rel_coord[:, :, 0] *= features.shape[2]
                    rel_coord[:, :, 1] *= features.shape[3]
                    
                    rel_cell = cell_sizes.clone()
                    rel_cell[:, :, 0] *= features.shape[2]
                    rel_cell[:, :, 1] *= features.shape[3]

                    #print("fc_input : " + str(fc_input.shape))
                    pred = self.UltraSR_forward(q_feat.permute(1, 2, 0).contiguous(), rel_coord, rel_cell)                    
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
        if("2D" in self.opt['mode']):
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            
            for pred, area in zip(preds, areas):  
                ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

class MFFN_temporal(nn.Module):
    def __init__(self, opt):
        super(MFFN_temporal, self).__init__()
        self.continuous = True
        self.opt = opt
        num_input = opt['base_num_kernels'] + 3
        self.MFFN = nn.ModuleList([
            nn.Linear(num_input, 512),
            nn.SiLU(),
            nn.Linear(num_input+512, 256),
            nn.SiLU(),
            nn.Linear(num_input+256, 128),
            nn.SiLU(),
            nn.Linear(num_input+128, 64),
            nn.SiLU(),
            nn.Linear(num_input+64, 32),
            nn.SiLU(),
            nn.Linear(32, opt['num_channels'])
        ])
        
        self.apply(weights_init)

    def forward(self, features, locations, cell_sizes=None):
        lr_shape = features.shape
        
        context_vectors = F.grid_sample(features, locations.flip(-1).unsqueeze(0), 
            'bilinear', align_corners=False)
        context_vectors = torch.cat([context_vectors, 
            locations.flip(-1).permute(3, 0, 1, 2).unsqueeze(0)], dim=1)
        context_vectors = context_vectors.permute(0, 2, 3, 4, 1).contiguous()
        x = self.MFFN[1](self.MFFN[0](context_vectors))
        x = self.MFFN[3](self.MFFN[2](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[5](self.MFFN[4](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[7](self.MFFN[6](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[9](self.MFFN[8](torch.cat([x, context_vectors], dim=-1)))
        x = self.MFFN[10](x)
        return x[0]

class Shuffle(nn.Module):
    def __init__(self, opt):
        super(Shuffle, self).__init__()
        self.continuous = False
        self.opt = opt
        if("2D" in opt['mode']):
            conv_layer = nn.Conv2d
        else:
            conv_layer = nn.Conv3d
        n_dims = 2 if "2D" in opt['mode'] else 3

        self.first_conv = conv_layer(opt['base_num_kernels'], 
            opt['base_num_kernels']*(round(1/opt['spatial_downscale_ratio'])**n_dims),
            kernel_size=opt['kernel_size'], padding=1)
        self.second_conv = conv_layer(opt['base_num_kernels'], opt['base_num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        self.final_conv = conv_layer(opt['base_num_kernels'], opt['num_channels'],
        stride=opt['stride'],padding=2,kernel_size=5)

    def forward(self, x, *kwargs):
        b = x.shape[0]
        x = self.first_conv(x)
        sf = round(1/self.opt['spatial_downscale_ratio'])
        n_dims = 2 if "2D" in self.opt['mode'] else 3
        fact = sf**n_dims
        if("2D" in self.opt['mode']):
            x = x.contiguous().view(
                b, sf, sf, int(x.shape[1]/fact), x.shape[2], x.shape[3]
            )
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
            x = x.view(
                b, int(x.shape[1]), sf*x.shape[2], sf*x.shape[4]
            )            
        else:
            x = x.contiguous().view(
                b, sf, sf, sf, int(x.shape[1]/fact), x.shape[2], x.shape[3], x.shape[4]
            )
            x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
            x = x.view(
                b, int(x.shape[1]/fact), sf*x.shape[2], sf*x.shape[4], sf*x.shape[6]
            )
        x = self.second_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.final_conv(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.0):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

class ImplicitMLP(nn.Module):
    def __init__(self, opt):
        super(ImplicitMLP, self).__init__()        
        self.opt = opt
        
        nodes_per_layer = 128
        self.n_dims = 2 if "2D" in opt['mode'] else 3
        self.model = nn.ModuleList([
            nn.Linear(self.n_dims, nodes_per_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(nodes_per_layer, opt["num_channels"])
        ])
        self.model = nn.Sequential(*self.model)
        self.apply(weights_init)

    def forward(self, locations):
        return self.model(locations)

class PositionalEncoding(nn.Module):
    def __init__(self, opt):
        super(PositionalEncoding, self).__init__()        
        self.opt = opt
        self.n_dims = 2 if "2D" in opt['mode'] else 3
        self.L = opt['num_positional_encoding_terms']
        self.L_terms = torch.arange(0, opt['num_positional_encoding_terms'], 
            device=opt['device'], dtype=torch.float32).repeat_interleave(2*self.n_dims)
        self.L_terms = torch.pow(2, self.L_terms) * pi
        #self.phase_shift = torch.rand(self.L_terms.shape, 
            #device=opt['device'])

    def forward(self, locations):
        repeats = len(list(locations.shape)) * [1]
        repeats[-1] = self.L*2
        locations = locations.repeat(repeats)
        
        locations = locations * self.L_terms# + self.phase_shift
        if(self.n_dims == 2):
            locations[..., 0::4] = torch.sin(locations[..., 0::4])
            locations[..., 1::4] = torch.sin(locations[..., 1::4])
            locations[..., 2::4] = torch.cos(locations[..., 2::4])
            locations[..., 3::4] = torch.cos(locations[..., 3::4])
        else:
            locations[..., 0::6] = torch.sin(locations[..., 0::6])
            locations[..., 1::6] = torch.sin(locations[..., 1::6])
            locations[..., 2::6] = torch.sin(locations[..., 2::6])
            locations[..., 3::6] = torch.cos(locations[..., 3::6])
            locations[..., 4::6] = torch.cos(locations[..., 4::6])
            locations[..., 5::6] = torch.cos(locations[..., 5::6])
        return locations

class VolumeToFC(nn.Module):
    def __init__(self, opt):
        super(VolumeToFC, self).__init__()
        self.mode = opt['mode']

    def forward(self, x):
        if "2D" in self.mode:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
        return x

class FCToVolume(nn.Module):
    def __init__(self, opt):
        super(FCToVolume, self).__init__()
        self.mode = opt['mode'] 

    def forward(self, x):
        if "2D" in self.mode:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 4, 1, 2, 3)
        return x

class Reshaper(nn.Module):
    def __init__(self, s):
        super(Reshaper, self).__init__()
        self.s = s

    def forward(self, x):
        return x.reshape(self.s)

class ACORN(nn.Module):
    def __init__(self, nodes_per_layer, opt):
        super(ACORN, self).__init__()        
        self.opt = opt

        self.n_dims = 2 if "2D" in opt['mode'] else 3
        self.last_layer_output = opt['feat_grid_channels']*opt['feat_grid_x']*opt['feat_grid_y']
        if "3D" in opt['mode']:
            self.last_layer_output *= opt['feat_grid_z']
        
        self.feat_grid_shape = [1, self.opt['feat_grid_channels'], 
            self.opt['feat_grid_x'], self.opt['feat_grid_y']]        
        if "3D" in self.opt['mode']:
            self.feat_grid_shape.append(self.opt['feat_grid_z'])

        self.positional_encoder = PositionalEncoding(opt)

        self.feature_encoder = nn.Sequential(
            #nn.Linear(self.n_dims, nodes_per_layer),
            nn.Linear(2*self.n_dims*opt['num_positional_encoding_terms'], nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, self.last_layer_output)
        )
        
        self.feature_decoder = nn.Sequential(
            VolumeToFC(opt),
            nn.Linear(opt['feat_grid_channels'], 64),
            nn.ReLU(),
            nn.Linear(64, opt['num_channels']),
            FCToVolume(opt)
        )

        self.apply(weights_init)
    
    def forward(self, global_coordinates, block_shape, mult = 1.0):
        # Todo
        return global_coordinates
        
    def forward_batch(self, global_coordinates, blocks, mult = 1.0):
        global_coordinates = self.positional_encoder(global_coordinates)
        # global coordinates will be B x n_dims
        all_block_vals = []
        shape = [global_coordinates.shape[0], self.opt['feat_grid_channels'], 
            self.opt['feat_grid_x'], self.opt['feat_grid_y']]
        if "3D" in self.opt['mode']:
            shape.append(self.opt['feat_grid_z'])

        vals = self.feature_encoder(global_coordinates)
        vals = vals.reshape(shape)
        for i in range(global_coordinates.shape[0]):

            b_vals = F.interpolate(vals[i:i+1], size=blocks[i].shape[2:], 
                mode='bilinear' if "2D" in self.opt['mode'] else "trilinear", 
                align_corners=False)

            b_vals = self.feature_decoder(b_vals)

            all_block_vals.append(b_vals * mult)
        
        return all_block_vals

class HierarchicalACORN(nn.Module):
    def __init__(self, opt):
        super(HierarchicalACORN, self).__init__()        
        self.opt = opt
        self.models = nn.ModuleList([ACORN(int(2**(6.54)), opt)])
        self.errors = [1]
        self.residual = None
        
    def add_model(self, opt):
        self.models.append(ACORN(int(2**(len(self.models)*0.5+6)), opt))

    def forward(self, block, block_position, model_no):        
        block_output = self.models[model_no](block_position, block, self.errors[-1])
        return block_output

    def calculate_block_errors(self, octree, error_func):
        blocks, block_positions = octree.depth_to_blocks_and_block_positions(
                octree.max_depth)
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.errors[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                block_outputs[b] += self.residual[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3]]
            else:
               block_outputs[b] +=  self.residual[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3],                    
                    blocks[b].start_position[2]:\
                        blocks[b].start_position[2]+block_outputs[b].shape[4]]
            error = error_func(block_outputs[b], blocks[b].data())
            blocks[b].error = error.item()

    def get_full_img(self, octree):
        blocks, block_positions = octree.depth_to_blocks_and_block_positions(
                octree.max_depth)
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        out = torch.zeros_like(octree.data, device=self.opt['device'])
        out += self.residual
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.errors[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3]] += block_outputs[b]
            else:
               out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3],                    
                    blocks[b].start_position[2]:\
                        blocks[b].start_position[2]+block_outputs[b].shape[4]] += block_outputs[b]
        return out

class GenericModel(nn.Module):
    def __init__(self, opt):
        super(GenericModel, self).__init__()
        self.opt = opt
        print(opt['mode'])
        if("implicit" in self.opt['mode']):
            self.model = ImplicitMLP(opt)
        else:
            if(self.opt['feat_model'] == "RDN"):
                self.feature_extractor = RDN(opt)
            elif(self.opt['feat_model'] == "RDN_skip"):
                self.feature_extractor = RDN_skip(opt)
            elif(self.opt['feat_model'] == "RRDN"):
                self.feature_extractor = RRDN(opt)
            elif(self.opt['feat_model'] == "UNet"):
                self.feature_extractor = UNet(opt)
            elif(self.opt['feat_model'] == "Identity"):
                self.feature_extractor = Identity(opt)
            
            if(self.opt['upscale_model'] == "LIIF"):
                self.upscaling_model = LIIF(opt)
            elif(self.opt['upscale_model'] == "LIIF_skip"):
                self.upscaling_model = LIIF_skip(opt)
            elif(self.opt['upscale_model'] == "LIIF_skip_posonly"):
                self.upscaling_model = LIIF_skip_posonly(opt)
            elif(self.opt['upscale_model'] == "LIIF_skip_res"):
                self.upscaling_model = LIIF_skip_res(opt)
            elif(self.opt['upscale_model'] == "MFFN"):
                self.upscaling_model = MFFN(opt)
            elif(self.opt['upscale_model'] == "UltraSR"):
                self.upscaling_model = UltraSR(opt)
            elif(self.opt['upscale_model'] == "MFFN_temporal"):
                self.upscaling_model = MFFN_temporal(opt)
            elif(self.opt['upscale_model'] == "Shuffle"):
                self.upscaling_model = Shuffle(opt)
        
    def forward(self, lr, locations=None, cell_sizes=None):
        if("implicit" in self.opt['mode']):
            points = self.model(lr)
        else:
            features = self.feature_extractor(lr)
            points = self.upscaling_model(features, locations, cell_sizes)
        return points