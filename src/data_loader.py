# Definition the EgoCap dataset (small version)
import torch
import torchvision
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
import random
from collections import OrderedDict 

from utils_transformations import warp_image, change_appearance
from utils_transformations import c_src_list, c_dst_list

# utility dictionary that can move tensor values between devices via the 'to(device)' function
class DeviceDict(dict):
    # following https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
    def __init__(self, *args):
      super(DeviceDict, self).__init__(*args)
    def to(self, device):
      dd = DeviceDict() # return shallow copy
    
      for k,v in self.items():
          if torch.is_tensor(v):
              dd[k] = v.to(device)
          else:
              dd[k] = v
      return dd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(Dataset).__init__();
        
        if dataset == "EgoCap":
            data_file = dataset+'_nth10.hdf5'
            with h5py.File( data_file, 'r') as hf:
                self.imgs  = torch.from_numpy(hf['img'][...])
        elif dataset == "sample_DeepFashion":
            data_file = dataset+'.h5'
            with h5py.File( data_file, 'r') as hf:
                self.imgs  = torch.from_numpy(hf['img'][...]).permute(0,3,1,2)
        elif dataset == "CUB":
            data_file = dataset+'.h5'
            with h5py.File( data_file, 'r') as hf:
                self.imgs  = torch.from_numpy(hf['img'][...]).permute(0,3,1,2)
                self.poses = torch.from_numpy(hf["pose"][...])
                self.train = hf["train"][...]
                self.test = hf["test"][...]
        elif dataset == "MAFL":
            data_file = dataset+'.h5'
            with h5py.File( data_file, 'r') as hf:
                self.imgs  = torch.from_numpy(hf['img'][...]).permute(0,3,1,2)
                self.poses = torch.from_numpy(hf["pose"][...])

                    
        print("Loading dataset to memory, can take some seconds")
        

        print(".. done loading")
        self.dataset = dataset
        self.mean, self.std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = 1/self.std),
                                               transforms.Normalize(mean = -self.mean, std = [ 1., 1., 1. ])])

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        
      index = np.random.randint(6)
      rand = random.uniform(0.7,1)
      c_src = c_src_list[index] * random.uniform(0.7,1)
      c_dst = c_dst_list[index] * rand
        
            
      angle = np.random.randint(120) - 60
        
      img = self.imgs[idx]
#       # randomly flip image from left to right
#       if (np.random.randint(2) == 0):
#           img = torch.fliplr(torch.flip(img, [1,2]))
      
      orig_img = self.normalize(img.float()/255)
        
      trans_img = warp_image((img).permute(1,2,0), c_src , c_dst, angle, 255) #trans_img is numpy array
      trans_img = torch.from_numpy(trans_img).permute(2,0,1)
      app_trans_img = change_appearance(trans_img.permute(1,2,0))
      trans_img = self.normalize(trans_img.float()/255)

      denormal_trans_img = warp_image((img).permute(1,2,0), c_src , c_dst, angle, 255)
      denormal_trans_img = torch.from_numpy(denormal_trans_img).permute(2,0,1)
      denormal_trans_img = (denormal_trans_img.float()/255)

      app_img = change_appearance(img.permute(1,2,0))
        
      if self.dataset == "CUB" or self.dataset == "MAFL":
          if (np.random.randint(1) == 0):     
              sample = DeviceDict(
                      {'img': orig_img,
                       "denorm_img": img.float()/255,
                       "trans_img": DeviceDict({
                          "img": trans_img,
                      }),
                      "app_trans_img": DeviceDict({
                              "img": self.normalize(app_trans_img)
                      }),
                      "app_img": DeviceDict({
                          "img": self.normalize(app_img)
                      }), 
                      "pose": self.poses[idx], 
                      "denorm_trans_img": denormal_trans_img})
          else:
              sample = DeviceDict(
                      {'img': trans_img,
                       "denorm_img": denormal_trans_img ,
                       "trans_img": DeviceDict({
                          "img": orig_img,
                      }),
                      "app_trans_img": DeviceDict({
                          "img": self.normalize(app_img)
                      }), 
                      "app_img": DeviceDict({
                          "img": self.normalize(app_trans_img)
                      }), 
                      "pose": self.poses[idx], 
                      "denorm_trans_img": img.float()/255})
            
      else:
        ## Swapping technique: randomly change orig image with trans_img
          if (np.random.randint(1) != 0):     
              sample = DeviceDict(
                          {'img': trans_img,
                           "denorm_img": denormal_trans_img,
                           "trans_img": DeviceDict({
                              "img": orig_img
                           }),
                          "app_img": DeviceDict({
                              "img": self.normalize(app_trans_img)
                          }), 
                          "app_trans_img": DeviceDict({
                              "img": self.normalize(app_img)
                          }), 
                          "denorm_trans_img": (img.float()/255)})
          else:
              sample = DeviceDict(
                          {'img': orig_img,
                           "denorm_img": img.float()/255,
                           "trans_img": DeviceDict({
                              "img": trans_img
                           }),
                          "app_img": DeviceDict({
                              "img": self.normalize(app_img)
                          }), 
                          "app_trans_img": DeviceDict({
                              "img": self.normalize(app_trans_img)
                          }), 
                          "denorm_trans_img": denormal_trans_img})            
    
      return sample
