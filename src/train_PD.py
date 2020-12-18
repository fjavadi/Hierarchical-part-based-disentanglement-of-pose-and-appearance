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

def train(dataset, data_loader, decoder, int_network, num_epochs, accumulation_steps, filename):
    
    lr=0.001
    
    params = list(decoder.parameters()) + list(int_network.parameters()) 
    optimizer = torch.optim.Adam(params, lr)
    optimizer.zero_grad()
    
    
    fig=plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    axes=fig.subplots(3,3)

    losses = [0]
    reconst_losses = [0]
    p_losses = [0]
    perceptual_loss = VGGPerceptualLoss()
    pixelwise_loss = 0 

    for e in range(num_epochs):
      train_iter = iter(data_loader)
      for i in range(len (data_loader)):

          batch_cpu = next(train_iter)
          
        # create blobby image for reconstructing the original img
          blobby_img, heatmap_app, heatmap_trans, pose_app = create_blobby_img(int_network, part_img = batch_cpu["app_img"], color_img = batch_cpu["trans_img"], denorm_color_img = batch_cpu["denorm_trans_img"])

          reconst_img = decoder(blobby_img)
          src_img = batch_cpu["img"].cuda()  

          # optimize network
          reconst_loss = torch.nn.functional.mse_loss(reconst_img, src_img) 
          p_loss = perceptual_loss(reconst_img, src_img)
          loss = reconst_loss + 0.1 * p_loss
          wandb.log({"loss": loss.item()})
          loss.backward()
            
           ### accumulating the gradient 
          if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                          # Now we can do an optimizer step
                optimizer.zero_grad()
                
                #save losses
                losses.append(loss.item())
                p_losses.append(p_loss.item())
                reconst_losses.append(reconst_loss.item())


          reconst_img = reconst_img.detach().cpu()
          blobby_img = blobby_img.detach().cpu()
          pixelwise_loss += torch.sum((reconst_img.cuda() - src_img)**2) / (src_img.shape[3] * src_img.shape[2])
 
          if (i%10 == 0):

              #visulaize
              for ax in axes: 
                    for a in ax:
                        a.cla()
                        a.axis('off')
                
            
              plotDotOneImage(pose_app[0], dataset.denormalize(batch_cpu['img'][0]), marker= "o", ax=axes[0,0])
#               plotPoseOnImage([], dataset.denormalize(batch_cpu['img'][0]), ax=axes[0,0])
              plotPoseOnImage([], dataset.denormalize(batch_cpu["trans_img"]['img'][0]), ax=axes[0,1])

              plotPoseOnImage([], dataset.denormalize(batch_cpu["trans_img"]['img'][0]), ax=axes[0,1])
              axes[0,2].imshow(torchvision.transforms.ToPILImage()(dataset.denormalize(batch_cpu["app_img"]["img"][0])))

              axes[1,0].imshow(torchvision.transforms.ToPILImage()(blobby_img[0]))
              axes[1,2].imshow(torchvision.transforms.ToPILImage()(np.clip(dataset.denormalize(reconst_img[0]), 0 ,1)))
              axes[2,2].imshow(torchvision.transforms.ToPILImage()(dataset.denormalize(batch_cpu["app_trans_img"]["img"][0])))

              plotPoseOnImage([], heatmap2image(heatmap_app[0]), ax=axes[2,0])
              plotPoseOnImage([], heatmap2image(heatmap_trans[0]), ax=axes[2,1])
#               axes[2,2].imshow(torchvision.transforms.ToPILImage()(pixelwise_loss.cpu()))
            

                
              # save results in wandb
              resulted_images = []
              resulted_images.append(wandb.Image(dataset.denormalize(batch_cpu['img'][0]))) 
              resulted_images.append(wandb.Image(heatmap2image(heatmap_app[0])))
              resulted_images.append(wandb.Image(heatmap2image(heatmap_trans[0])))
              resulted_images.append(wandb.Image(np.clip(dataset.denormalize(reconst_img[0]), 0 ,1)))
              resulted_images.append(wandb.Image(blobby_img[0]))  
    
              wandb.log({"results": resulted_images})    

          #clear output window and diplay updated figure
              display.clear_output(wait=True)
              display.display(plt.gcf())
                
          #save models      
          if (i%500==1):
            torch.save(decoder.state_dict(), "saved_models/decoder_"+filename+".pt")
            torch.save(int_network.state_dict(), "saved_models/int_network_"+filename+".pt")
          
          print("Epoch {}, iteration {} of {} ({} %), loss={}, {}, {}".format(e, i, len(data_loader), 100*i//len(data_loader), losses[-1], p_losses[-1], reconst_losses[-1]))
                
    plt.close('all')
    return 


# get the gaussians from part_img and colors from color_img to create blobby image
def create_blobby_img(int_network, part_img, color_img, denorm_color_img):
    
            #Appearance stream
          pred_raw_color_img = int_network(color_img)
          pred_integral_color_img = integral_heatmap_layer(pred_raw_color_img) 
          gauss_parts_color_img = estimate_Gaussian(pred_integral_color_img["pose_2d"], pred_integral_color_img["cov_matrix"], pred_integral_color_img["probabilitymap"].shape)
          part_apps = calculate_appearance(denorm_color_img, gauss_parts_color_img)

          # Pose Stream
          pred_raw_part_img = int_network(part_img)
          pred_integral_part_img = integral_heatmap_layer(pred_raw_part_img) 
          #gaussians are between 0-1
          gauss_parts = estimate_Gaussian(pred_integral_part_img["pose_2d"], pred_integral_part_img["cov_matrix"], pred_integral_part_img["probabilitymap"].shape)
        
        
        # Reconst Stream: combine guassians by their appearances
          x = part_apps.reshape(part_apps.shape[0], part_apps.shape[1], part_apps.shape[2], 1, 1)\
            .expand(part_apps.shape[0],part_apps.shape[1],part_apps.shape[2],gauss_parts.shape[2],gauss_parts.shape[3])\
                                                * (gauss_parts.reshape(gauss_parts.shape[0],gauss_parts.shape[1],1,gauss_parts.shape[2],gauss_parts.shape[3])\
                                                .expand(gauss_parts.shape[0],gauss_parts.shape[1],3,gauss_parts.shape[2],gauss_parts.shape[3])) 
         # create the blobby image
          blobby_img= torch.max(x,dim=1)[0]

          return blobby_img.cuda(), pred_integral_part_img['probabilitymap'].cpu(), pred_integral_color_img['probabilitymap'].cpu(), pred_integral_part_img["pose_2d"].cpu()