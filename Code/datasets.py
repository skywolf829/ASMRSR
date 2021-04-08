import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import h5py
import torch.nn.functional as F
from Code.utility_functions import AvgPool2D, AvgPool3D

class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        self.item_names = []
        self.subsample_dist = 1
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):
            self.item_names.append(filename)
            self.num_items += 1
        self.item_names.sort()

    def __len__(self):
        return self.num_items

    def resolution(self):
        return self.resolution

    def scale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "magnitude"):
            d *= (1/self.max_mag)
        elif (self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[:,i] -= self.channel_mins[i]
                d[:,i] /= (self.channel_maxs[i] - self.channel_mins[i])
                d[:,i] -= 0.5
                d[:,i] *= 2
        return d

    def unscale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[:, i] *= 0.5
                d[:, i] += 0.5
                d[:, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[:, i] += self.channel_mins[i]
        elif(self.opt['scaling_mode'] == "magnitude"):
            d *= self.max_mag
        return d

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist
        
    def __getitem__(self, index):
        if(self.opt['load_data_at_start']):
            data = self.items[index]
        else:

            #print("trying to load " + str(self.item_names[index]) + ".h5")
            f = h5py.File(os.path.join(self.opt['data_folder'], self.item_names[index]), 'r')
            x_start = 0
            x_end = self.opt['x_resolution']
            y_start = 0
            y_end = self.opt['y_resolution']
            if(self.opt['mode'] == "3D"):
                z_start = 0
                z_end = self.opt['z_resolution']
                if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                    z_start = torch.randint(self.opt['z_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                    z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

            if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
                y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
            if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
                x_start = torch.randint(self.opt['x_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist
            
            if(self.opt['downsample_mode'] == "average_pooling"):
                #print("converting " + self.item_names[index] + " to tensor")
                if(self.opt['mode'] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end,
                        z_start:z_end])
                elif(self.opt['mode'] == "2D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end])
                f.close()
                
                if(self.subsample_dist > 1):
                    if(self.opt["mode"] == "3D"):
                        data = AvgPool3D(data.unsqueeze(0), self.subsample_dist)[0]
                    elif(self.opt['mode'] == "2D"):
                        data = AvgPool2D(data.unsqueeze(0), self.subsample_dist)[0]
                    
            elif(self.opt['downsample_mode'] == "subsampling"):
                if(self.opt["mode"] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist,
                        z_start:z_end:self.subsample_dist])
                elif(self.opt['mode'] == "2D"):       
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist])                 
                f.close()
            else:
                if(self.opt["mode"] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end,
                        z_start:z_end])
                elif(self.opt['mode'] == "2D"):   
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist])
                f.close()
                data = F.interpolate(data.unsqueeze(0), scaling_factor=float(1/self.subsample_dist), 
                mode=self.opt['downsample_mode'], align_corners=True)[0]
            #print("converted " + self.item_names[index] + ".h5 to tensor")


        '''
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                data[i] -= self.channel_mins[i]
                data[i] *= (1 / (self.channel_maxs[i] - self.channel_mins[i]))
                data[i] -= 0.5
                data[i] *= 2
        elif(self.opt['scaling_mode'] == "magnitude"):
            data *= (1 / self.max_mag)
        '''

        if(self.opt['mode'] == "3Dto2D"):
            data = data[:,:,:,int(data.shape[3]/2)]

        #data = np2torch(data, "cpu")
        #print("returning " + str(index) + " data")
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data = torch.flip(data,[3])

        return data