# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley
# Date:    10/11/2019
# Purpose: Model for solving RPM problems

# IMPORTS ----------------------------------------------------------------------------------------------------------- #

import os
import glob
import numpy as np
import sys
import torch
import random
from   torch.utils.data import Dataset

# SCRIPT ------------------------------------------------------------------------------------------------------------ #

#By default, the dataloader will pad data to the maximum number of slots, before shuffling over those slots. 
#   This is to allow the model to learn to use all channels, else testing on more objects than seen in training will likely fail.
#   However, shuffling before padding might increase performance when training and testing over the entire dataset, particularly 
#   when using smaller percentages of the data to train.

rpm_folders = {'cs': "center_single", 
               'io': "in_center_single_out_center_single", 
               'ud': "up_center_single_down_center_single",
               'lr': "left_center_single_right_center_single",
               'd4': "distribute_four",
               'd9': "distribute_nine",
               '4c': "in_distribute_four_out_center_single",
               '*' : '*'}

class dataset(Dataset):
    def __init__(self, args, mode, rpm_types):
        self.root_dir = args.path
        self.img_size = args.img_size
        self.set      = args.dataset
        self.model    = args.model
        self.n_s      = args.trn_n if mode!="test" else args.tst_n
        self.mode     = mode
        self.percent  = args.percent if mode!="test" else 100
        self.objects  = args.objects
        self.shuffle_first = args.shuffle_first
        
        if self.set=="pgm":
            file_names = [f for f in os.listdir(self.root_dir) if mode in f]
            random.shuffle(file_names)
            self.file_names = file_names[:int(len(file_names)*self.percent/100)]
            
        else:
            file_names = [[f for f in glob.glob(os.path.join(self.root_dir, rpm_folders[t], "*.npz")) if mode in f] for t in rpm_types]
            [random.shuffle(sublist) for sublist in file_names]
            file_names = [item for sublist in file_names for item in sublist[:int(len(sublist)*self.percent/100)]]
            file_names = [f for f in file_names if "small" in f]
            self.file_names = file_names
        
    def __len__(self):
        return len(self.file_names)
    
    def shuffle(self, obj, pos):
        frames_o = []
        frames_p = []
        for f in zip(obj,pos):
            idx = torch.randperm(obj.size(1))
            frames_o.append(f[0][idx])
            frames_p.append(f[1][idx])
        obj = torch.stack(frames_o)
        pos = torch.stack(frames_p)
        return obj,pos
    
    def pad(self, obj, pos, dim):
        if self.n_s > obj.size(1):
            pad_o = torch.zeros((16, self.n_s-obj.size(1), dim, dim))
            pad_p = torch.zeros((16, self.n_s-obj.size(1), 3))
            obj   = torch.cat((obj, pad_o), dim=1)
            pos   = torch.cat((pos, pad_p), dim=1)
        return obj,pos
    
    def __getitem__(self, idx):
        if self.set=="pgm":
            #Read.
            data_path = self.root_dir+'/'+self.file_names[idx]
            data = np.load(data_path)
            images = data["image"]
            target = data["target"]
            
            #Shuffle choices.
            if self.mode=="train":
                context = images[:8]
                choices = images[8:]
                indices = list(range(8))
                np.random.shuffle(indices)
                new_target = indices.index(target)
                new_choices = choices[indices]
                images = np.concatenate((context, new_choices))
                target = new_target
            
            #Return tensors.
            return torch.tensor(images, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
        
        else:
            data_path = self.file_names[idx]
            data = np.load(data_path)
            target = data["target"]
            img = data["image"]
    
            #If model is Rel-AIR, load objects and position/scale data.
            if self.model=='Rel-AIR':
                if self.objects=='attention':
                    obj = torch.tensor(data["x_att"], dtype=torch.float32)
                    dim = obj.size(-1)
                elif self.objects=='reconstruction':
                    obj = torch.tensor(data["obj"], dtype=torch.float32)
                    dim = int(obj.size(-1)**0.5)
                    obj = obj.view(16, obj.size(1), dim, dim)
                elif self.objects=='all':
                    obj1 = torch.tensor(data["x_att"], dtype=torch.float32)
                    obj2 = torch.tensor(data["obj"], dtype=torch.float32)
                    dim  = obj1.size(-1)
                    obj  = torch.cat((obj1, obj2.view(16, -1, dim, dim)), dim=1)
                else:
                    print('Object type unrecognised.')
                    sys.exit(1)
                    
                pos = torch.tensor(data["latent"], dtype=torch.float32)[:,:,50:]
                pos = pos.repeat(1, 2, 1) if self.objects=='all' else pos
                y   = torch.tensor(target, dtype=torch.long)
                
                #Normalise scale and pos data to (0,1). These parameters are based on the prior mean+sd used in AIR.
                scale = pos[:, :, 0] #Range = (1, 4.5)
                scale = (scale-1)/(4.5-1)
                x_pos = pos[:, :, 1] #Range = (-2.5, 2.5)
                x_pos = (x_pos+2.5)/(2.5+2.5)
                y_pos = pos[:, :, 2] #Range = (-2.5, 2.5)
                y_pos = (y_pos+2.5)/(2.5+2.5)
                pos = torch.stack((scale, x_pos, y_pos), dim=-1)
                
                #Store the actual number of slots (pre-padding), to un-pad later.
                n = obj.size(1)
                    
                #First, pad to N_S if data has less than the full number of objects per frame.
                #   Then, shuffle each frame of the current RPM along the object axis, independently.
                #   Alternatively, shuffle before padding, if requested.
                if self.shuffle_first:
                    obj,pos = self.shuffle(obj, pos)
                    obj,pos = self.pad(obj, pos, dim)
                else:
                    obj,pos = self.pad(obj, pos, dim)
                    obj,pos = self.shuffle(obj, pos)

                #Shuffle the choice order and reassign target accordingly.
                if self.mode=="train":
                    context_o, context_p = obj[:8],pos[:8]
                    choices_o, choices_p = obj[8:],pos[8:]
                    idx = list(range(8))
                    np.random.shuffle(idx)
                    obj = torch.cat((context_o, choices_o[idx]), dim=0)
                    pos = torch.cat((context_p, choices_p[idx]), dim=0)
                    y = torch.tensor(idx.index(target), dtype=torch.long)

                return (obj,pos,n,img),y
                
            #Else (if model is ResNet or Rel-Base), load images.
            else:
                images = data["image"]
                
                #Shuffle choices to a) avoid exploiting any statistical bias in the dataset, and b) mitigate overfitting.
                if self.mode=="train":
                    context = images[:8]
                    choices = images[8:]
                    idx = list(range(8))
                    np.random.shuffle(idx)
                    images = np.concatenate((context, choices[idx]))
                    target = idx.index(target)        
                images = torch.tensor(images, dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.long)
                
                return images, target

# END SCRIPT -------------------------------------------------------------------------------------------------------- #