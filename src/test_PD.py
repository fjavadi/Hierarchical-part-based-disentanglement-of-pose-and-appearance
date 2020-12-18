import matplotlib.pyplot as plt
from IPython import display
import torch 
import wandb

import torchvision
import numpy as np


from utils_plots import plotPoseOnImage, heatmap2image
from utils import integral_heatmap_layer, estimate_Gaussian, transform_heatmap, calculate_appearance

def test(dataset, data_loader, decoder, int_network):

    
    ## Evaluation on the pose transfer task
    fig=plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    axes=fig.subplots(2,3)

    # 0(first image in the batch) is pose source and 1(second image in the batch) is app source
    data_iter = iter(data_loader)
    
    for i in range(len(data_loader)):
        batch_cpu = next(data_iter)
        batch_gpu = batch_cpu.to('cuda')

        pred_raw = int_network(batch_gpu)
        pred_integral = integral_heatmap_layer(pred_raw) # note, this function must be differentiable
    
        ## pose source first img 

        
        gauss_parts = estimate_Gaussian(pred_integral["pose_2d"], pred_integral["cov_matrix"], pred_integral["probabilitymap"].shape)
        part_apps = calculate_appearance(batch_gpu["denorm_img"], gauss_parts)
        part_apps = torch.roll(part_apps, 1, 0)

        x = part_apps.reshape(part_apps.shape[0], part_apps.shape[1], part_apps.shape[2], 1, 1)\
                                .expand(part_apps.shape[0],part_apps.shape[1],part_apps.shape[2],gauss_parts.shape[2],gauss_parts.shape[3])\
                                *\
        (gauss_parts.reshape(gauss_parts.shape[0],gauss_parts.shape[1],1,gauss_parts.shape[2],gauss_parts.shape[3])\
                                                .expand(gauss_parts.shape[0],gauss_parts.shape[1],3,gauss_parts.shape[2],gauss_parts.shape[3]))  
        blobby_img= torch.max(x,dim=1)[0]
        reconst_img = decoder(blobby_img)


        #visulaize
        pred_cpu = pred_integral.to('cpu')
        blobby_img = blobby_img[0].detach().cpu()

        for ax in axes: 
            for a in ax:
                a.cla()

        plotPoseOnImage([], dataset.denormalize(batch_cpu['img'][0].cpu()), ax=axes[0,0])
        plotPoseOnImage([], dataset.denormalize(batch_cpu['img'][1].cpu()), ax=axes[0,1])
        axes[1,0].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img[0].detach().cpu()), 0 ,1)))
        axes[1,1].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img[1].detach().cpu()), 0 ,1)))

        #wandb
        resulted_images = []
        resulted_images.append(wandb.Image(dataset.denormalize(batch_cpu['img'][0]))) 
        resulted_images.append(wandb.Image(heatmap2image(pred_integral["probabilitymap"][0])))

        resulted_images.append(wandb.Image(dataset.denormalize(batch_cpu['img'][1]))) 
        resulted_images.append(wandb.Image(heatmap2image(pred_integral["probabilitymap"][1]))) 
        
        resulted_images.append(wandb.Image(blobby_img[0])) 
        resulted_images.append(wandb.Image(blobby_img[1])) 

        resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img[0].detach().cpu()), 0 ,1)))     
        resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img[1].detach().cpu()), 0 ,1)))             

        wandb.log({"results": resulted_images})    

        #   clear output window and diplay updated figure
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close('all')
    return
