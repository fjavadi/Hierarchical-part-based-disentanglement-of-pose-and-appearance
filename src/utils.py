import torch
import torchvision
import torch.nn as nn

from scipy.special import softmax
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt

from utils_transformations import warp_image
from data_loader import DeviceDict

def integral_heatmap_layer(dict):
    # compute coordinate matrix 
    heatmap = dict['heatmap'].cuda() #heatmap-size= N*K*H*W
    ##Normalize heatmaps
    h_norm =torch.zeros(heatmap.shape).cuda()
    tmp = heatmap- ((heatmap.max(2)[0]).max(2)[0]).reshape(heatmap.shape[0],heatmap.shape[1],1,1).expand((heatmap.shape))
    sum_matrix = torch.sum(torch.exp(tmp),(2,3)).reshape(heatmap.shape[0],heatmap.shape[1],1,1).expand((heatmap.shape))
    h_norm = torch.exp(tmp)/sum_matrix
    
    #Find centers
    x=torch.linspace(0,1,heatmap.shape[3],requires_grad=False).reshape(1,1,1,heatmap.shape[3]).expand(heatmap.shape).cuda()
    pose_x=torch.sum(x*h_norm,(2,3))
    y=torch.linspace(0,1,heatmap.shape[2],requires_grad=False).reshape(1,1,heatmap.shape[2],1).expand(heatmap.shape).cuda()
    pose_y=torch.sum(y*h_norm,(2,3))
    pose=torch.stack((pose_x,pose_y),2)
    
    ##find standard deviation
    x_dist= x - pose_x.reshape(heatmap.shape[0],heatmap.shape[1],1,1).expand(heatmap.shape) # x and pose_x are 0-1
    var_x= torch.sum((x_dist**2)*h_norm,(2,3)) / (heatmap.shape[2]*heatmap.shape[3])
    y_dist= y - pose_y.reshape(heatmap.shape[0],heatmap.shape[1],1,1).expand(heatmap.shape)
    var_y=torch.sum((y_dist**2)*h_norm,(2,3)) / (heatmap.shape[2]*heatmap.shape[3])
    
    cov_xy= torch.sum((x_dist)*(y_dist)*h_norm, (2,3))/ (heatmap.shape[2]*heatmap.shape[3])
    
    cov_mat = torch.zeros(heatmap.shape[0],heatmap.shape[1],2,2).cuda()
    cov_mat[:,:,0,0] = var_x*heatmap.shape[3]*heatmap.shape[3]
    cov_mat[:,:,0,1] = cov_xy*heatmap.shape[2]*heatmap.shape[3]
    cov_mat[:,:,1,0] = cov_xy*heatmap.shape[2]*heatmap.shape[3]
    cov_mat[:,:,1,1] = var_y *heatmap.shape[2]*heatmap.shape[2]
    
    return DeviceDict({'probabilitymap': h_norm.cpu(), 'pose_2d': pose, "cov_matrix": cov_mat})

def estimate_Gaussian(mean,cov,size):

    xy= torch.zeros((size[2],size[3],1,2)).cuda()
    for y in range(size[2]):
        for x in range(size[3]):
            xy[y][x]=torch.FloatTensor([[x/size[3],y/size[2]]])
    # resizing        
    xy= xy.reshape(1,1,size[2],size[3],1,2).expand(size[0],size[1],size[2],size[3],1,2)
    mean= mean.reshape(size[0],size[1],1,1,1,2).expand(size[0],size[1],size[2],size[3],1,2)
    
    # formula : exp(-0.5 * (x-M)E^(x-M).T ) --->> M = mean E^ = invers of the covariance matrix
    inv_cov = torch.inverse(cov)
    inv_cov= inv_cov.reshape(size[0],size[1],1,1,2,2).expand(size[0],size[1],size[2],size[3],2,2)
    d= torch.matmul(xy-mean, inv_cov)
    d= torch.matmul(d, torch.transpose(xy,4,5) - torch.transpose(mean,4,5))
    gauss_parts= torch.exp(-0.5* d) #parts=4d tensor n*k*h*w all values between 0 and 1 each h*w is a gaussian filter   
    
    return gauss_parts 

def calculate_appearance(img, heatmap):
    ### the function for encoding the appearances 
    # img = n*3*h*2 , heatmap = n*18*h*w
    img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2], img.shape[3]).expand(img.shape[0], heatmap.shape[1], img.shape[1], img.shape[2], img.shape[3]).cuda()
    heatmap = heatmap.reshape(img.shape[0],img.shape[1],1,img.shape[3],img.shape[4]).expand(img.shape).cuda()
    appearance = torch.sum(img * heatmap, dim=(3,4))/torch.sum(heatmap,dim=(3,4)).cuda()

    return appearance #n*k*3

def transform_heatmap(dict, c_src, c_dst, angle):
    ### this function apply a transfromation on a single heatmap.  
    
    heatmap = dict['heatmap'] #heatmap-size= N*K*H*W
    shape = heatmap.shape
    rst_heatmap = torch.zeros(shape)
    c_src = c_src.cpu().numpy()
    c_dst = c_dst.cpu().numpy()
    angle = angle.cpu().numpy()
    
    # choose transformation
    for i in range(shape[0]):
        for j in range (shape[1]):
            trans_img = warp_image(heatmap[i][j].unsqueeze(0).expand(3, shape[2], shape[3]).permute(1,2,0).detach().cpu()\
                       , c_src[i], c_dst[i], angle[i]) 
            rst_heatmap[i][j] = torch.from_numpy(trans_img).permute(2,0,1).cuda()[0]
    
    return DeviceDict({'heatmap': rst_heatmap}) 

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg = torchvision.models.vgg16(pretrained=True).cuda()
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss