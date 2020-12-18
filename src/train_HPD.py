import matplotlib.pyplot as plt
from IPython import display
import time
from scipy import signal
import numpy as np
import wandb
import io

import torch 
import torchvision

from utils_plots import plotPoseOnImage, heatmap2image, plotDotOneImage
from utils import integral_heatmap_layer, estimate_Gaussian, transform_heatmap, calculate_appearance, VGGPerceptualLoss

torch.set_printoptions(profile="default")

def train(dataset, data_loader, decoder, int_network, num_epochs, accumulation_steps, k, filename):
    
    lr=0.001
    
    params = list(decoder.parameters()) + list(int_network.parameters()) 
    optimizer = torch.optim.Adam(params, lr)
    optimizer.zero_grad()
    perceptual_loss = VGGPerceptualLoss()

    
    fig=plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    axes=fig.subplots(3,3)

    losses = []
    reconst_losses = []
    hrc_losses = []
    p_losses = []

    for e in range(num_epochs):
      train_iter = iter(data_loader)
      for i in range(len (data_loader)):

          batch_cpu = next(train_iter)
          
        # create blobby image for reconstructing original img
          blobby_img1, blobby_img2, heatmap_app, heatmap_trans, pose_app = create_blobby_img(int_network, part_img = batch_cpu, color_img = batch_cpu, denorm_color_img = batch_cpu["denorm_img"], k = k)
          

          reconst_img1 = decoder(blobby_img1)
          reconst_img2 = decoder(blobby_img2)
#         reconst_img3= decoder(blobby_img3) ---> for the case of 3 levels of hierarchy

          batch_gpu = batch_cpu["img"].cuda()

          # l2
          reconst_loss = 0.5 * (torch.nn.functional.mse_loss(reconst_img1,  batch_gpu) +\
                         torch.nn.functional.mse_loss(reconst_img2,  batch_gpu))
#                          torch.nn.functional.mse_loss(reconst_img3,  batch_gpu) 
         # perceptual_loss
          p_loss = 0.5 * (perceptual_loss(reconst_img1, batch_gpu)  +\
                          perceptual_loss(reconst_img2, batch_gpu))
                                
          # hierarchical loss                     
          hrc_loss = torch.nn.functional.mse_loss(pose_app[:,:k,:], pose_app[:,np.arange(k, 3*k, 2),:]) +\
                     torch.nn.functional.mse_loss(pose_app[:,:k,:], pose_app[:,np.arange(k+1, 3*k, 2),:])
                       ## For the third level
#                      torch.nn.functional.mse_loss(pose_app[:,k:3*k,:], pose_app[:,np.arange(3*k, 7*k, 2),:]) +\
#                      torch.nn.functional.mse_loss(pose_app[:,k:3*k,:], pose_app[:,np.arange(3*k+1, 7*k, 2),:])

          loss = reconst_loss +  0.1 * hrc_loss + 1 * p_loss
            
        
          # wandb
          wandb.log({"loss": loss.item()})
          wandb.log({"hrc_loss": hrc_loss.item()})
          wandb.log({"rec_loss": reconst_loss.item()+ p_loss.item() })
        
    
          # optimize network
          loss.backward()
          if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
                losses.append(loss.item())
                hrc_losses.append(hrc_loss.item())

          reconst_img1 = reconst_img1.detach().cpu()
          reconst_img2 = reconst_img2.detach().cpu()
#           reconst_img3 = reconst_img.detach().cpu()
          blobby_img1 = blobby_img1.detach().cpu()
          blobby_img2 = blobby_img2.detach().cpu()
#           blobby_img3 = blobby_img3.detach().cpu()


          if (i%10 == 0):
              #visulaize
              for ax in axes: 
                    for a in ax:
                        a.cla()
                        a.axis('off')



              plotPoseOnImage([], dataset.denormalize(batch_cpu['img'][0]), ax=axes[0,0])
              plotPoseOnImage([], dataset.denormalize(batch_cpu["trans_img"]['img'][0]), ax=axes[0,1])
              axes[0,2].imshow(torchvision.transforms.ToPILImage()(dataset.denormalize(batch_cpu["app_img"]["img"][0])))

              axes[1,0].imshow(torchvision.transforms.ToPILImage()(blobby_img1[0]))
              axes[1,1].imshow(torchvision.transforms.ToPILImage()(blobby_img2[0]))
#               axes[1,2].imshow(torchvision.transforms.ToPILImage()(blobby_img3[0]))
                    
              axes[2,0].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img1[0]), 0 ,1)))
              axes[2,1].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img2[0]), 0 ,1)))
#               axes[2,2].imshow(torchvision.c.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img3[0]), 0 ,1)))
     
              #wandb visualization
              resulted_images = []

              resulted_images.append(wandb.Image(dataset.denormalize(batch_cpu['img'][0])))
              ## heatmaps of the first level of hierarchy
              resulted_images.append(wandb.Image(heatmap2image(heatmap_app[0,np.arange(0,k),:])))
               ## heatmaps of the second level of hierarchy
              resulted_images.append(wandb.Image(heatmap2image(heatmap_app[0,np.arange(k, 3*k),:])))
              ## reconstructed image of the first level of hierarchy                 
              resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img1[0]), 0 ,1)))  
              ## reconstructed image of the second level of hierarchy                 
              resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img2[0]), 0 ,1)))  
              ## blobby image of the first level of hierarchy                 
              resulted_images.append(wandb.Image(blobby_img1[0])) 
              ## blobby image of the second level of hierarchy                 
              resulted_images.append(wandb.Image(blobby_img2[0]))

              wandb.log({"results": resulted_images})    

          #clear output window and diplay updated figure
              display.clear_output(wait=True)
              display.display(plt.gcf())
                
          #save models
          if (i%500==1):
            torch.save(decoder.state_dict(), "saved_models/decoder_"+filename+".pt")
            torch.save(int_network.state_dict(), "saved_models/int_network_"+filename+".pt")
            
          if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
              print("Epoch {}, iteration {} of {} ({} %), loss={}, {}".format(e, i, len(data_loader), 100*i//len(data_loader), losses[-1], hrc_losses[-1]))
            
    plt.close('all')
    return 


# get the gaussians from part_img and colors from color_img to create blobby image
def create_blobby_img(int_network, part_img, color_img, denorm_color_img, k):

            # Appearance Stream
          pred_raw_color_img = int_network(color_img)
          pred_integral_color_img = integral_heatmap_layer(pred_raw_color_img) # note, this function must be differentiable
          gauss_parts_color_img = estimate_Gaussian(pred_integral_color_img["pose_2d"], pred_integral_color_img["cov_matrix"], pred_integral_color_img["probabilitymap"].shape)
          part_apps = calculate_appearance(denorm_color_img, gauss_parts_color_img)

         # Pose Stream
          pred_raw_part_img = int_network(part_img)
          pred_integral_part_img = integral_heatmap_layer(pred_raw_part_img) # note, this function must be differentiable
          gauss_parts = estimate_Gaussian(pred_integral_part_img["pose_2d"], pred_integral_part_img["cov_matrix"], pred_integral_part_img["probabilitymap"].shape)

        
            # Reconst Stream
          x = part_apps.reshape(part_apps.shape[0], part_apps.shape[1], part_apps.shape[2], 1, 1)\
                                .expand(part_apps.shape[0],part_apps.shape[1],part_apps.shape[2],gauss_parts.shape[2],gauss_parts.shape[3])\
                                *\
        (gauss_parts.reshape(gauss_parts.shape[0],gauss_parts.shape[1],1,gauss_parts.shape[2],gauss_parts.shape[3])\
                                                .expand(gauss_parts.shape[0],gauss_parts.shape[1],3,gauss_parts.shape[2],gauss_parts.shape[3])) 
    
          ### create blobby images for each level of hierarchy
          blobby_img1 = torch.max(x[:, :k, :, :], dim=1)[0] #-----> first level of hierarchy
          blobby_img2 = torch.max(x[:, k:3*k, :, :], dim=1)[0] #-----> second level of hierarchy
#           blobby_img3 = torch.max(x[:, 3*k:7*k, :, :], dim=1)[0] #---> third level of hierarchy

          return blobby_img1.cuda(), blobby_img2.cuda(),  pred_integral_part_img['probabilitymap'], pred_integral_color_img['probabilitymap'], pred_integral_part_img["pose_2d"].cpu()