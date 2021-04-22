import imageio
import os
import numpy as np
import torch
import torch.nn as nn
from math import log10
import math
import numbers
from torch.nn import functional as F
from torch.nn import Parameter
from matplotlib.pyplot import cm
import time
import pickle
from math import exp
from typing import Dict, List, Tuple, Optional
import argparse

def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def make_residual_weight_grid(real_lr, hr_coords, mode):    
    lr_coord = make_coord(real_lr.shape[2:], device=real_lr.device,
        flatten=False)

    if(mode == "2D"):
        lr_coord = lr_coord.permute(2, 0, 1).\
            unsqueeze(0).expand(real_lr.shape[0], 2, *real_lr.shape[2:])
    else:
        lr_coord = lr_coord.permute(3, 0, 1, 2).\
            unsqueeze(0).expand(real_lr.shape[0], 3, *real_lr.shape[2:])
    q_coord = F.grid_sample(
        lr_coord, hr_coords.flip(-1).unsqueeze(0),
        mode='nearest', align_corners=False)[0]
    #print("Q coord: " + str(q_coord.shape))
    rel_coord = hr_coords - q_coord.permute(1, 2, 0)

    for c in range(2, len(real_lr.shape)):
        if mode == "2D":
            rel_coord[:,:,c-2] *= real_lr.shape[c]
        else:
            rel_coord[:,:,:,c-2] *= real_lr.shape[c]
            
    rel_coord = torch.norm(rel_coord, dim=-1)
    rel_coord *= (1/(2**0.5))

    rel_coord = rel_coord.unsqueeze(0).unsqueeze(0)
    if(mode == "2D"):
        rel_coord = rel_coord.expand(real_lr.shape[0], real_lr.shape[1], -1, -1)
    else:
        rel_coord = rel_coord.expand(real_lr.shape[0], real_lr.shape[1], -1, -1, -1)
    return rel_coord

def make_coord(shape, device, flatten=True):
    """ 
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        r = 1 / (n)
        left = -1.0
        right = 1.0
        seq = left + r + (2 * r) * torch.arange(0, n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if(flatten):
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(vol, flatten=True):
    """ Convert the image/volume to coord-val pairs.
        vol: Tensor, (B, C, H, W) or (B, C, H, W, L)
    """
    coord = make_coord(vol.shape[2:], vol.device, flatten)
    vals = vol.view(vol.shape[1], -1).permute(1, 0)
    return coord, vals

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.01)

def VoxelShuffle(t):
    input_view = t.contiguous().view(
        1, 2, 2, 2, int(t.shape[1]/8), t.shape[2], t.shape[3], t.shape[4]
    )
    shuffle_out = input_view.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    out = shuffle_out.view(
        1, int(t.shape[1]/8), 2*t.shape[2], 2*t.shape[3], 2*t.shape[4]
    )
    return out

def create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn):
    bnl = batchnorm_layer(num_kernels)
    bnl.apply(weights_init)
    if(use_sn):
        bnl = SpectralNorm(bnl)
    return bnl

def create_conv_layer(conv_layer, in_chan, out_chan, kernel_size, stride, padding, use_sn):
    c = conv_layer(in_chan, out_chan, 
                    kernel_size, stride, 0)
    c.apply(weights_init)
    if(use_sn):
        c = SpectralNorm(c)
    return c

def MSE(x, GT):
    return ((x-GT)**2).mean()

def PSNR(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    p = 20 * torch.log10(max_diff) - 10*torch.log10(MSE(x, GT))
    return p

def relative_error(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    val = np.abs(GT-x).max() / max_diff
    return val

def pw_relative_error(x, GT):
    val = np.abs(np.abs(GT-x) / GT)
    return val.max()

def gaussian(window_size : int, sigma : float) -> torch.Tensor:
    gauss : torch.Tensor = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x \
        in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size : torch.Tensor, channel : int) -> torch.Tensor:
    _1D_window : torch.Tensor = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window : torch.Tensor = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window : torch.Tensor = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1 : torch.Tensor, img2 : torch.Tensor, window : torch.Tensor, 
window_size : torch.Tensor, channel : int, size_average : Optional[bool] = True):
    mu1 : torch.Tensor = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 : torch.Tensor = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq : torch.Tensor = mu1.pow(2)
    mu2_sq : torch.Tensor = mu2.pow(2)
    mu1_mu2 : torch.Tensor = mu1*mu2

    sigma1_sq : torch.Tensor = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq : torch.Tensor = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 : torch.Tensor = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 : float = 0.01**2
    C2 : float= 0.03**2

    ssim_map : torch.Tensor = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ans : torch.Tensor = torch.Tensor([0])
    if size_average:
        ans = ssim_map.mean()
    else:
        ans = ssim_map.mean(1).mean(1).mean(1)
    return ans

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel).to("cuda:2")
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel).to("cuda:2")

    mu1_sq = mu1.pow(2).to("cuda:2")
    mu2_sq = mu2.pow(2).to("cuda:2")

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel).to("cuda:2") - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel).to("cuda:2") - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel).to("cuda:2") - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    #ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    mu1_sq += mu2_sq
    mu1_sq += C1

    sigma1_sq += sigma2_sq
    sigma1_sq += C2

    mu1_sq *= sigma1_sq

    mu1_mu2 *= 2
    mu1_mu2 += C1

    sigma12 *= 2
    sigma12 += C2

    mu1_mu2 *= sigma12

    mu1_mu2 /= mu1_sq

    if size_average:
        return mu1_mu2.mean()
    else:
        return mu1_mu2.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def to_img(input : torch.Tensor, mode : str, colormap = True):
    if(mode == "2D"):
        img = input[0].clone().detach()
        img -= img.min()
        img *= (1/img.max()+1e-6)
        if(colormap and img.shape[0] == 1):
            img = cm.coolwarm(img[0].cpu().numpy())
            #img = np.transpose(img, (2, 0, 1))
            img = (255*img).astype(np.uint8)
        else:
            img *= 255
            img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    elif(mode == "3D"):
        img = input[0,:,:,:,int(input.shape[4]/2)].clone()
        img -= img.min()        
        img *= (1/img.max()+1e-6)
        if(colormap and img.shape[0] == 1):
            img = cm.coolwarm(img[0].cpu().numpy())
            #img = np.transpose(img, (2, 0, 1))
            img = (255*img).astype(np.uint8)
        else:
            img *= 255
            img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    #print(img.shape)
    return img

def bilinear_interpolate(im, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[3]-1)
    y1 = torch.clamp(y1, 0, im.shape[3]-1)
    
    Ia = im[0, :, x0, y0 ]
    Ib = im[0, :, x1, y0 ]
    Ic = im[0, :, x0, y1 ]
    Id = im[0, :, x1, y1 ]
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    return Ia*wa + Ib*wb + Ic*wc + Id*wd

def trilinear_interpolate(im, x, y, z, device, periodic=False):

    if(device == "cpu"):
        dtype = torch.float
        dtype_long = torch.long
    else:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    z0 = torch.floor(z).type(dtype_long)
    z1 = z0 + 1
    
    if(periodic):
        x1_diff = x1-x
        x0_diff = 1-x1_diff  
        y1_diff = y1-y
        y0_diff = 1-y1_diff
        z1_diff = z1-z
        z0_diff = 1-z1_diff

        x0 %= im.shape[2]
        y0 %= im.shape[3]
        z0 %= im.shape[4]

        x1 %= im.shape[2]
        y1 %= im.shape[3]
        z1 %= im.shape[4]
        
    else:
        x0 = torch.clamp(x0, 0, im.shape[2]-1)
        x1 = torch.clamp(x1, 0, im.shape[2]-1)
        y0 = torch.clamp(y0, 0, im.shape[3]-1)
        y1 = torch.clamp(y1, 0, im.shape[3]-1)
        z0 = torch.clamp(z0, 0, im.shape[4]-1)
        z1 = torch.clamp(z1, 0, im.shape[4]-1)
        x1_diff = x1-x
        x0_diff = x-x0    
        y1_diff = y1-y
        y0_diff = y-y0
        z1_diff = z1-z
        z0_diff = z-z0
    
    c00 = im[0,:,x0,y0,z0] * x1_diff + im[0,:,x1,y0,z0]*x0_diff
    c01 = im[0,:,x0,y0,z1] * x1_diff + im[0,:,x1,y0,z1]*x0_diff
    c10 = im[0,:,x0,y1,z0] * x1_diff + im[0,:,x1,y1,z0]*x0_diff
    c11 = im[0,:,x0,y1,z1] * x1_diff + im[0,:,x1,y1,z1]*x0_diff

    c0 = c00 * y1_diff + c10 * y0_diff
    c1 = c01 * y1_diff + c11 * y0_diff

    c = c0 * z1_diff + c1 * z0_diff
    return c   
    
def lagrangian_transport(VF, x_res, y_res, time_length, ts_per_sec, device):
    #x = torch.arange(-1, 1, int(VF.shape[2] / x_res), dtype=torch.float32).unsqueeze(1).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)

    x = torch.arange(0, VF.shape[2], int(VF.shape[2] / x_res), dtype=torch.float32).view(1, -1).repeat([x_res, 1])
    x = x.view(1,x_res, y_res)
    #y = torch.arange(-1, 1, int(VF.shape[3] / y_res), dtype=torch.float32).unsqueeze(0).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)
    y = torch.arange(0, VF.shape[3], int(VF.shape[3] / y_res), dtype=torch.float32).view(-1, 1).repeat([1, y_res])
    y = y.view(1, x_res,y_res)
    particles = torch.cat([x, y],axis=0)
    particles = torch.reshape(particles, [2, -1]).transpose(0,1)
    particles = particles.to(device)
    #print(particles)
    particles_over_time = []
    
    
    for i in range(0, time_length * ts_per_sec):
        particles_over_time.append(particles.clone())
        start_t = time.time()
        flow = bilinear_interpolate(VF, particles[:,0], particles[:,1])
        particles[:] += flow[0:2, :].permute(1,0) * (1 / ts_per_sec)
        particles[:] += torch.tensor(list(VF.shape[2:])).to(device)
        particles[:] %= torch.tensor(list(VF.shape[2:])).to(device)
    particles_over_time.append(particles)
    
    return particles_over_time

def lagrangian_transport3D(VF, x_res, y_res, z_res, 
time_length, ts_per_sec, device, periodic=False):
    #x = torch.arange(-1, 1, int(VF.shape[2] / x_res), dtype=torch.float32).unsqueeze(1).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)

    x = torch.arange(0, VF.shape[2], VF.shape[2] / x_res, 
    dtype=torch.float32).view(-1, 1, 1).repeat([1, y_res, z_res])
    x = x.view(1,x_res,y_res, z_res)
    y = torch.arange(0, VF.shape[3], VF.shape[3] / y_res, 
    dtype=torch.float32).view(1, -1, 1).repeat([x_res, 1, z_res])
    y = y.view(1,x_res,y_res, z_res)
    z = torch.arange(0, VF.shape[4], VF.shape[4] / z_res, 
    dtype=torch.float32).view(1, 1, -1).repeat([x_res, y_res, 1])
    z = z.view(1,x_res,y_res, z_res)

    particles = torch.cat([x, y, z],axis=0)
    particles = torch.reshape(particles, [3, -1]).transpose(0,1)
    particles = particles.to(device)
    particles_over_time = []
        
    for i in range(0, time_length * ts_per_sec):
        particles_over_time.append(particles.clone())
        start_t = time.time()
        flow = trilinear_interpolate(VF, particles[:,0], particles[:,1], particles[:,2], device)
        particles[:] += flow[:, :].permute(1,0) * (1 / ts_per_sec)
        if(periodic):
            particles[:] += torch.tensor(list(VF.shape[2:])).to(device)
            particles[:] %= torch.tensor(list(VF.shape[2:])).to(device)
        else:
            particles[:] = torch.clamp(particles, 0, VF.shape[2])
    particles_over_time.append(particles)
    
    return particles_over_time

def viz_streamlines(frame, streamlines, name, color):
    arr = np.zeros(frame.shape)
    arrs = []

    for i in range(len(streamlines)):
        arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
        for j in range(streamlines[i].shape[0]):
            arr[0, :, int(streamlines[i][j, 0]), int(streamlines[i][j, 1])] = color
    arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
    imageio.mimwrite(name + ".gif", arrs)
    return arrs

def streamline_distance(pl1, pl2):
    d = torch.norm(pl1[0] - pl2[0], dim=1).sum()
    for i in range(1, len(pl1)):
        d += torch.norm(pl1[i] - pl2[i], dim=1).sum()
    return d

def streamline_err_volume(real_VF, rec_VF, res, ts_per_sec, time_length, device, periodic=False):
    
    x = torch.arange(0, real_VF.shape[2], 1, 
    dtype=torch.float32).view(-1, 1, 1).repeat([1, real_VF.shape[3], real_VF.shape[4]])
    x = x.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    y = torch.arange(0, real_VF.shape[3], 1, 
    dtype=torch.float32).view(1, -1, 1).repeat([real_VF.shape[2], 1, real_VF.shape[4]])
    y = y.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    z = torch.arange(0, real_VF.shape[4], 1, 
    dtype=torch.float32).view(1, 1, -1).repeat([real_VF.shape[2], real_VF.shape[3], 1])
    z = z.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    particles_real = torch.cat([x, y, z],axis=3).to(device)
    particles_rec = particles_real.clone()
    
    transport_loss_volume = torch.zeros([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4]], device=device)
    
    
    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            flow_real = trilinear_interpolate(real_VF, 
            particles_real.reshape([-1, 3])[:,0] % real_VF.shape[2], 
            particles_real.reshape([-1, 3])[:,1] % real_VF.shape[3], 
            particles_real.reshape([-1, 3])[:,2] % real_VF.shape[4], device, periodic = periodic)
            
            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec.reshape([-1, 3])[:,0] % rec_VF.shape[2], 
            particles_rec.reshape([-1, 3])[:,1] % rec_VF.shape[3], 
            particles_rec.reshape([-1, 3])[:,2] % rec_VF.shape[4], device, periodic = periodic)

            particles_real += flow_real.transpose(0,1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 3])
            particles_rec += flow_rec.transpose(0,1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 3])
            transport_loss_volume += torch.norm(particles_real-particles_rec, dim=3)
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2], 
            device, periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2], 
            device, periodic = periodic)

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss_volume += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).transpose(0, 1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4]])
    
    #print("t_init: %0.07f, t_interp: %0.05f, t_add: %0.07f, t_total: %0.07f" % (t_create_particles, t_interp, t_add, time.time()-t_start))
    return transport_loss_volume / (time_length * ts_per_sec)

def streamline_loss3D(real_VF, rec_VF, x_res, y_res, z_res, ts_per_sec, time_length, device, periodic=False):
    
    t_start = time.time()
    t = time.time()
    particles_real = torch.rand([3,x_res*y_res*z_res]).to(device).transpose(0,1)
    particles_real[:,0] *= real_VF.shape[2]
    particles_real[:,1] *= real_VF.shape[3]
    particles_real[:,2] *= real_VF.shape[4]
    particles_rec = particles_real.clone()
    t_create_particles = time.time() - t
    t_add = 0
    t_interp = 0
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))

    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            t = time.time()
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[:,0] % real_VF.shape[2], 
            particles_real[:,1] % real_VF.shape[3], 
            particles_real[:,2] % real_VF.shape[4], device, periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[:,0] % rec_VF.shape[2], 
            particles_rec[:,1] % rec_VF.shape[3], 
            particles_rec[:,2] % rec_VF.shape[4], device, periodic = periodic)
            t_interp += time.time() - t

            t = time.time()
            particles_real += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec += flow_rec.permute(1,0) * (1 / ts_per_sec)

            transport_loss += torch.norm(particles_real -particles_rec, dim=1).mean()
            t_add += time.time() - t
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2], 
            device, periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2], 
            device, periodic = periodic)

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).mean()
    
    #print("t_init: %0.07f, t_interp: %0.05f, t_add: %0.07f, t_total: %0.07f" % (t_create_particles, t_interp, t_add, time.time()-t_start))
    return transport_loss / (time_length * ts_per_sec)

def adaptive_streamline_loss3D(real_VF, rec_VF, error_volume, n, octtree_levels,
ts_per_sec, time_length, device, periodic=False):
    
    e_total = error_volume.sum()
    particles_real = torch.zeros([3, n], device=device)
    current_spot = 0
    octtreescale = 3
    #for octtreescale in range(octtree_levels):
    domain_size_x = int((1.0 / (2**octtreescale)) * error_volume.shape[0])
    domain_size_y = int((1.0 / (2**octtreescale)) * error_volume.shape[1])
    domain_size_z = int((1.0 / (2**octtreescale)) * error_volume.shape[2])
    
    for x_start in range(0, error_volume.shape[0], domain_size_x):
        for y_start in range(0, error_volume.shape[1], domain_size_y):
            for z_start in range(0, error_volume.shape[2], domain_size_z):
                error_in_domain = error_volume[x_start:x_start+domain_size_x,
                y_start:y_start+domain_size_y,z_start:z_start+domain_size_z].sum() / e_total
                n_particles_in_domain = int(n * error_in_domain)
                
                particles_real[:,current_spot:current_spot+n_particles_in_domain] = \
                torch.rand([3,n_particles_in_domain])

                particles_real[0,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_x
                particles_real[1,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_y
                particles_real[2,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_z

                particles_real[0,current_spot:current_spot+n_particles_in_domain] += \
                x_start
                particles_real[1,current_spot:current_spot+n_particles_in_domain] += \
                y_start
                particles_real[2,current_spot:current_spot+n_particles_in_domain] += \
                z_start
                current_spot += n_particles_in_domain
                '''
                for i in range(n_particles_in_domain):
                    particles_real[:,current_spot] = torch.rand([3]) * domain_size
                    particles_real[0,current_spot] += x_start
                    particles_real[1,current_spot] += y_start
                    particles_real[2,current_spot] += z_start
                    current_spot += 1
                '''
    particles_real[:,current_spot:] = torch.rand([3, particles_real.shape[1]-current_spot])
    particles_real[0,current_spot:] *= error_volume.shape[0]
    particles_real[1,current_spot:] *= error_volume.shape[1]
    particles_real[2,current_spot:] *= error_volume.shape[2]
        
    particles_real = particles_real.transpose(0,1)
    particles_rec = particles_real.clone()
    
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))
    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[:,0] % real_VF.shape[2], 
            particles_real[:,1] % real_VF.shape[3], 
            particles_real[:,2] % real_VF.shape[4], device)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[:,0] % rec_VF.shape[2], 
            particles_rec[:,1] % rec_VF.shape[3], 
            particles_rec[:,2] % rec_VF.shape[4], device)

            particles_real += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec += flow_rec.permute(1,0) * (1 / ts_per_sec)

            transport_loss += torch.norm(particles_real-particles_rec, dim=1).mean()
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2], device)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2], device)

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss += torch.norm(particles_real[indices]-particles_rec[indices], dim=1).mean()
    return transport_loss / (time_length * ts_per_sec)
    
def sample_adaptive_streamline_seeds(error_volume, n, device):
    e_total = error_volume.sum()
    particles = torch.zeros([3, n], device=device)
    current_spot = 0
    octtreescale = 3
    #for octtreescale in range(octtree_levels):
    domain_size_x = int((1.0 / (2**octtreescale)) * error_volume.shape[0])
    domain_size_y = int((1.0 / (2**octtreescale)) * error_volume.shape[1])
    domain_size_z = int((1.0 / (2**octtreescale)) * error_volume.shape[2])
    
    for x_start in range(0, error_volume.shape[0], domain_size_x):
        for y_start in range(0, error_volume.shape[1], domain_size_y):
            for z_start in range(0, error_volume.shape[2], domain_size_z):
                error_in_domain = error_volume[x_start:x_start+domain_size_x,
                y_start:y_start+domain_size_y,z_start:z_start+domain_size_z].sum() / e_total
                n_particles_in_domain = int(n * error_in_domain)
                
                
                particles[:,current_spot:current_spot+n_particles_in_domain] = \
                torch.rand([3,n_particles_in_domain])
                
                particles[0,current_spot:current_spot+n_particles_in_domain] *= domain_size_x
                particles[1,current_spot:current_spot+n_particles_in_domain] *= domain_size_y
                particles[2,current_spot:current_spot+n_particles_in_domain] *= domain_size_z

                particles[0,current_spot:current_spot+n_particles_in_domain] += x_start
                particles[1,current_spot:current_spot+n_particles_in_domain] += y_start
                particles[2,current_spot:current_spot+n_particles_in_domain] += z_start
                current_spot += n_particles_in_domain

    particles[:,current_spot:] = torch.rand([3, particles.shape[1]-current_spot])
    particles[0,current_spot:] *= error_volume.shape[0]
    particles[1,current_spot:] *= error_volume.shape[1]
    particles[2,current_spot:] *= error_volume.shape[2]

    particles = particles.type(torch.LongTensor).transpose(0,1)
    particles[:,0] = torch.clamp(particles[:,0], 0, error_volume.shape[0]-1)
    particles[:,1] = torch.clamp(particles[:,1], 0, error_volume.shape[1]-1)
    particles[:,2] = torch.clamp(particles[:,2], 0, error_volume.shape[2]-1)

    particle_volume = torch.zeros(error_volume.shape).type(torch.FloatTensor).to(device)
    for i in range(particles.shape[0]):
        particle_volume[particles[i,0],particles[i,1],particles[i,2]] += 1
    
    return particle_volume

def streamline_loss2D(real_VF, rec_VF, x_res, y_res, ts_per_sec, time_length, device, periodic=False):
    x = torch.arange(0, real_VF.shape[2], real_VF.shape[2] / x_res, 
    dtype=torch.float32).view(-1, 1).repeat([1, y_res])
    x = x.view(1,x_res,y_res)
    y = torch.arange(0, real_VF.shape[3], real_VF.shape[3] / y_res, 
    dtype=torch.float32).view(1, -1).repeat([x_res, 1])
    y = y.view(1,x_res,y_res)
    
    particles_real = torch.cat([x, y],axis=0)
    particles_real = torch.reshape(particles_real, [2, -1]).transpose(0,1)
    particles_real = particles_real.to(device)

    particles_rec = torch.cat([x, y],axis=0)
    particles_rec = torch.reshape(particles_rec, [2, -1]).transpose(0,1)
    particles_rec = particles_rec.to(device)
    
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))
    for i in range(0, time_length * ts_per_sec):
        indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
        (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
        (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
        (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) 
        
        flow_real = bilinear_interpolate(real_VF, 
        particles_real[indices,0], particles_real[indices,1])

        flow_rec = bilinear_interpolate(rec_VF, 
        particles_rec[indices,0], particles_rec[indices,1])

        particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
        particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
        print(indices.sum())
        if(periodic):
            particles_real[:] += torch.tensor(list(real_VF.shape[2:])).to(device)
            particles_real[:] %= torch.tensor(list(real_VF.shape[2:])).to(device)
            particles_rec[:] += torch.tensor(list(rec_VF.shape[2:])).to(device)
            particles_rec[:] %= torch.tensor(list(rec_VF.shape[2:])).to(device)
        else:
            #with torch.no_grad():
            #    particles_real = torch.clamp(particles_real, 0, real_VF.shape[2])
            #    particles_rec = torch.clamp(particles_rec, 0, rec_VF.shape[2])
            transport_loss += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).mean()
    return transport_loss / (time_length * ts_per_sec)

def toImg(vectorField, renorm_channels = False):
    vf = vectorField.copy()
    if(len(vf.shape) == 3):
        if(vf.shape[0] == 1):
            return cm.coolwarm(vf[0]).swapaxes(0,2).swapaxes(1,2)
        elif(vf.shape[0] == 2):
            vf += 1
            vf *= 0.5
            vf = vf.clip(0, 1)
            z = np.zeros([1, vf.shape[1], vf.shape[2]])
            vf = np.concatenate([vf, z])
            return vf
        elif(vf.shape[0] == 3):
            if(renorm_channels):
                for j in range(vf.shape[0]):
                    vf[j] -= vf[j].min()
                    vf[j] *= (1 / vf[j].max())
            return vf
    elif(len(vf.shape) == 4):
        return toImg(vf[:,:,:,int(vf.shape[3]/2)], renorm_channels)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def AvgPool2D(x : torch.Tensor, size : int):
    with torch.no_grad():
        kernel = torch.ones([size, size]).to(x.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, size, size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        out = F.conv2d(x, kernel, stride=size, padding=0, groups=x.shape[1])
    return out

def AvgPool3D(x : torch.Tensor, size: int):
    with torch.no_grad():
        kernel = torch.ones([size, size, size]).to(x.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, size, size, size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1, 1)
        out = F.conv3d(x, kernel, stride=size, padding=0, groups=x.shape[1])
    return out
