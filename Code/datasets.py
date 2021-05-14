import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import h5py
import torch.nn.functional as F
from utility_functions import AvgPool2D, AvgPool3D, make_coord

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
                if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution'] and self.opt['cropping_resolution'] != -1):
                    z_start = torch.randint(self.opt['z_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                    z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

            if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution'] and self.opt['cropping_resolution'] != -1):
                y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
            if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution'] and self.opt['cropping_resolution'] != -1):
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


class LocalImplicitDataset(torch.utils.data.Dataset):
    def __init__(self, opt):        
        self.opt = opt
        
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):
            self.item = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
        self.item = torch.tensor(self.item).unsqueeze(0)
        self.coords = make_coord(self.item.shape, "cpu")        

        self.item = self.item[0].flatten(1).transpose(0,1).contiguous()
        self.coords = self.coords[0].flatten(1).transpose(0,1).contiguous()

        self.num_items = self.item.shape[0]
        print("Dataset initialized")

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        return (self.coords[index], self.item[index])


class LocalTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        self.item_names = []
        extension = ""
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):
            self.item_names.append(filename.split(".")[0])
            extension = filename.split(".")[1]
            self.num_items += 1

        self.item_names.sort(key=int)
        for i in range(len(self.item_names)):
            filename = self.item_names[i] + "." + extension
            ts = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
            data =  torch.tensor(ts['data'], dtype=torch.float32)
            ts.close()
            self.items.append(data)
            print(filename)
        
        print("Finished dataset init")
        self.num_items -= opt['time_cropping_resolution']
        self.num_items += 1

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        
        x_start = 0
        x_end = self.opt['x_resolution']
        y_start = 0
        y_end = self.opt['y_resolution']

        x_start = torch.randint(self.opt['x_resolution'] - \
            self.opt['cropping_resolution'], [1]).item()
        x_end = x_start + self.opt['cropping_resolution']

        y_start = torch.randint(self.opt['y_resolution'] - \
            self.opt['cropping_resolution'], [1]).item()
        y_end = y_start + self.opt['cropping_resolution']

        items = []
        for i in range(index, index+self.opt['time_cropping_resolution']):
            items.append(self.items[i][:,x_start:x_end,y_start:y_end].unsqueeze(0))
        
        item = torch.cat(items, dim=0)

        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                item = torch.flip(item,[2])
            if(torch.rand(1).item() > 0.5):
                item = torch.flip(item,[3])

        # switch t c w h to c w h t
        return item.permute(1, 2, 3, 0)