import matplotlib.pyplot as plt
from IPython import display
import torch 
import torchvision
import numpy as np
import wandb


from utils_plots import plotPoseOnImage, heatmap2image
from utils import integral_heatmap_layer, estimate_Gaussian, transform_heatmap, calculate_appearance

def test(dataset, data_loader, decoder, int_network, k):
        
    pixelwise_loss = 0 
    
    fig=plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    axes=fig.subplots(2,3)

    # 0 is pose source and 1 is app source
    data_iter = iter(data_loader)
    
    for i in range(len(data_loader)):
        batch_cpu = next(data_iter)
        batch_gpu = batch_cpu.to('cuda')

        # Pose and Appearance Stream
        pred_raw = int_network(batch_gpu)
        pred_integral = integral_heatmap_layer(pred_raw) # note, this function must be differentiable
        gauss_parts= estimate_Gaussian(pred_integral["pose_2d"], pred_integral["cov_matrix"], pred_integral["probabilitymap"].shape)
        part_apps = calculate_appearance(batch_gpu["denorm_img"], gauss_parts)
        
        # Reconst Stream
        x = part_apps.reshape(part_apps.shape[0], part_apps.shape[1], part_apps.shape[2], 1, 1)\
                                .expand(part_apps.shape[0],part_apps.shape[1],part_apps.shape[2],gauss_parts.shape[2],gauss_parts.shape[3])\
                                *\
        (gauss_parts.reshape(gauss_parts.shape[0],gauss_parts.shape[1],1,gauss_parts.shape[2],gauss_parts.shape[3])\
                                                .expand(gauss_parts.shape[0],gauss_parts.shape[1],3,gauss_parts.shape[2],gauss_parts.shape[3])) 
    
        ### in the testing phase we reconstruct the image only from the final parts
        blobby_img = torch.max(x[:, k:3*k, :, :], dim=1)[0] #-----> parts of the final level of hierarchy
        reconst_img = decoder(blobby_img)

        pixelwise_loss += torch.sum((reconst_img.cuda() - batch_gpu["img"])**2) / (batch_gpu["img"].shape[3] * batch_gpu["img"].shape[2])

        #visulaize
        pred_cpu = pred_integral.to('cpu')
        blobby_img = blobby_img[0].detach().cpu()
        reconst_img = reconst_img[0].detach().cpu()
                
        if (i%10==0):
            for ax in axes: 
                for a in ax:
                    a.cla()
             
            plotPoseOnImage([], dataset.denormalize(batch_cpu['img'][0]), ax=axes[0,0])
            plotPoseOnImage([], dataset.denormalize(batch_cpu["trans_img"]['img'][0]), ax=axes[0,1])
            axes[0,2].imshow(torchvision.transforms.ToPILImage()(dataset.denormalize(batch_cpu["app_img"]["img"][0])))

            axes[1,0].imshow(torchvision.transforms.ToPILImage()(blobby_img))                    
            axes[1,1].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img), 0 ,1)))

            #wandb visualization
            resulted_images = []
            resulted_images.append(wandb.Image(dataset.denormalize(batch_cpu['img'][0])))
            ## heatmaps of the final level of hierarchy
            resulted_images.append(wandb.Image(heatmap2image(pred_integral["probabilitymap"][0,np.arange(k, 3*k),:])))
            resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img), 0 ,1)))  
            resulted_images.append(wandb.Image(blobby_img)) 

            wandb.log({"results": resulted_images})  

        #   clear output window and diplay updated figure
            display.clear_output(wait=True)
            display.display(plt.gcf())
            
    plt.close('all')
    print("rec_loss = ", pixelwise_loss/10 ) #---> 10 = len sample test set
    return
