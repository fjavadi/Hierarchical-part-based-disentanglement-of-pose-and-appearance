# plotting utility functions
import matplotlib.pyplot as plt
import torchvision
import torch


### Just for EgoCap dataset --- ignore for other datasets

# skeleton pose definition
joint_names = ['head', 'neck', 'left-shoulder', 'left-elbow', 'left-wrist', 'left-finger', 'right-shoulder', 'right-elbow', 'right-wrist', 'right-finger', 'left-hip', 'left-knee', 'left-ankle', 'left-toe', 'right-hip', 'right-knee', 'right-ankle', 'right-toe']
bones_ego_str = [('head', 'neck'), ('neck', 'left-shoulder'), ('left-shoulder', 'left-elbow'), ('left-elbow', 'left-wrist'), ('left-wrist', 'left-finger'), ('neck', 'right-shoulder'), ('right-shoulder', 'right-elbow'), ('right-elbow', 'right-wrist'), ('right-wrist', 'right-finger'), 
                 ('left-shoulder', 'left-hip'), ('left-hip', 'left-knee'), ('left-knee', 'left-ankle'), ('left-ankle', 'left-toe'), ('right-shoulder', 'right-hip'), ('right-hip', 'right-knee'), ('right-knee', 'right-ankle'), ('right-ankle', 'right-toe'), ('right-shoulder', 'left-shoulder'), ('right-hip', 'left-hip')]
bones_ego_idx = [(joint_names.index(b[0]),joint_names.index(b[1])) for b in bones_ego_str]

r"""Plots skeleton pose on a matplotlib axis.

        Args:
            ax (Axis): plt axis to plot
            pose_2d (FloatTensor): tensor of keypoints, of shape K x 2
            bones (list): list of tuples, each tuple defining the keypoint indices to be connected by a bone 
        Returns:
            Module: self
"""            
def plot_skeleton(ax, pose_2d, bones=bones_ego_idx, linewidth=2, linestyle='-'):
    cmap = plt.get_cmap('hsv')
    for bone in bones:
        color = cmap(bone[1] * cmap.N // len(joint_names)) # color according to second joint index
        ax.plot(pose_2d[bone,0], pose_2d[bone,1], linestyle, color=color, linewidth=linewidth)
        

r"""Plots list of skeleton poses and image.

        Args:
            poses (list): list of pose tensors to be plotted
            ax (Axis): plt axis to plot
            bones (list): list of tuples, each tuple defining the keypoint indices to be connected by a bone 
        Returns:
            Module: self
"""       
def plotPoseOnImage(poses, img, ax=plt):
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_size = torch.FloatTensor(img_pil.size)
    if type(poses) is not list:
      poses = [poses]
    linestyles = ['-', '--', '-.', ':']
    for i, p in enumerate(poses):
      pose_px = p*img_size
      plot_skeleton(ax, pose_px.numpy(),linestyle=linestyles[i%len(linestyles)])
    ax.imshow(img_pil)

def plotDotOneImage(poses, img, marker, ax=plt):
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_size = torch.FloatTensor(img_pil.size)
    poses = (poses * img_size).detach().cpu().numpy()
#     poses1 = (poses1 * img_size).detach().cpu().numpy()
    cmap = plt.get_cmap('hsv')
    for i in range(poses.shape[0]):
        color = cmap(i * cmap.N // len(joint_names)) # color according to second joint index
        print(poses[i][0], poses[i][1])
        ax.plot([poses[i][0]], [poses[i][1]],  marker=marker, markersize=15, color=color)
    ax.imshow(img_pil)
    return ax.imshow(img_pil)


r"""Converts a multi channel heatmap to an RGB color representation for display.

        Args:
            heatmap (tensor): of size C X H x W
        Returns:
            image (tensor): of size 3 X H x W
"""       
def heatmap2image(heatmap):
    C,H,W = heatmap.shape
    cmap = plt.get_cmap('hsv')
    img = torch.zeros(3,H,W).to(heatmap.device)
    for i in range(C):
        color = torch.FloatTensor(cmap(i * cmap.N // C)[:3]).reshape([-1,1,1]).to(heatmap.device)
        img = torch.max(img, color * heatmap[i]) # max in case of overlapping position of joints
    # heatmap and probability maps might have small maximum value. Normalize per channel to make each of them visible
    img_max, indices = torch.max(img,dim=-1,keepdim=True)
    img_max, indices = torch.max(img_max,dim=-2,keepdim=True)
    return img/img_max

