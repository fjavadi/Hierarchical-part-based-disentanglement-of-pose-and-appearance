# Hierarchical Part-based Disentanglement of Pose and Appearance

## Introduction
We propose a new method, called HPD (Hierarchical Part-based Disentanglement), for learning
structured object parts alongside with disentangling their spatial and appearance
factors. Training needs no annotations or prior knowledge on any of the factors
or object classes, and can be applied to any image dataset without any limitations.

Refer to the thesis for more details of our model and the results.


## Directory structure

The overall directory structure of your new project should look like as follows. You are welcome to add/remove folders or files, but make sure to keep the overall structure.

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- A subset of DeepFashion dataset including 100 images in a h5 format.
│   └── raw            <- 100 sample raw images from the DeepFashion datasset 
│
├── trained_models     <- Trained models (PD and HPD), created by `torch.save(model.state_dict(), ..)`.
││
├── notebooks          <- Jupyter notebooks for data_processing and training PD and HPD. 
│
├── reports            <- Master's Thesis
│   └── figures        <- Model's images
└── src                <- Source code
```

## Benchmarking

This section stores the trained models.

model  | encoder | decoder | lr | epochs| n_final_keypoints
--------|--------|--------|:------:|:------:|:------:
PD| [Pose Endcoder](https://drive.google.com/file/d/1FQPLKfILW-rEoXvLOEZG5Zm4YF6fXcKh/view?usp=sharing) | [Decoder](https://drive.google.com/file/d/1kNa6PtS_dVK-IqLaIicn9IqbGNQJ8zmJ/view?usp=sharing)| 0.001| 350| 15
HPD| [Pose Endcoder](https://drive.google.com/file/d/19Vhbhlw6hhIcNoECw57ze2cRDmhrbSMB/view?usp=sharing) | [Decoder](https://drive.google.com/file/d/1EO3XYN7dEO1QYF1N6NiUgGkn7ss7mslm/view?usp=sharing)| 0.001 | 200 | 30

** batch_size should always be greater than 32 (Accumulation gradient technique is used to allow using large batch_sizes)

*** The trained HPD model consists of two levels of hierarchy, 15 parts in the first level and 30 in the second. 


## Requirements

- Python = 3.7.3
- PyTorch = 1.6.0
- CUDA = 10.2 
- wandb = 0.9.6 (For saving results and visualization)

## Dataset
The code can be applied on any unlabled image dataset. data_processing notebook saves image datasets as hdf5 files. data_loader.py only accepts hdf5 files as its input. 
* A small scale dataset consistinig of 100 images from  the DeepFashion dataset is provided to test the pre-trained models and see some sample results. 

For new datasets, firstly you need to store it as a hdf5 dataset, then modify the data_loader.py file. 

## Models
The decoder is a U-net network. (Refere to the models.py), and the encoder is deeplabv3_resnet50.
To change the  architecture of networks, firstly you need to change the models.py and desing your model, then define the models in the PD and HPD notebooks and start the training. 

## Training

For training the models fram scratch, modify PD and HPD notebooks to not start from the saved models. If you don't change the notebooks, training will start from the pre-trained models. 

## Evaluate
- testPD file evaluates the PD model on the task of Pose and Appearance Transfer. It can be run by the testPD notebook.
- testHPD file tests the HPD model on the image reconstrcution task in terms of the pixel-wise loss. It can be run by the testHPD notebook.

## List of Parameters
parameter  | file | denotes 
--------|:------:|:------:
epochs | PD.ipynb, HPD.ipynb | number of training epochs
batch_size | PD.ipynb, HPD.ipynb | size of the training/testing batch
accumulation_steps | PD.ipynb, HPD.ipynb | number of iterations the gradient accumulates
n_app_features | PD.ipynb, HPD.ipynb | dimension of the appearance vectors (=3)
num_joints | PD.ipynb, HPD.ipynb | total number of parts predicted by encoder
k | HPD.ipynb | number of parts in the first layer of hierarchy

** For changing the weight of the perceptual loss and the hierarchical loss, refer to train_PD.py and train_HPD.py files. 

## Contact and Reference
- For more information, please contact fjavadi@cs.ubc.ca
- The PD model core idea comes from [Lorenz et al.](https://arxiv.org/abs/1903.06946) 
