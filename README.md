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
│   └── figures        
└── src                <- Source code
```

## Benchmarking

This section stores the trained models.

model  | encoder | decoder | lr 
--------|--------|--------|:------:
PD| [Pose Endcoder](https://drive.google.com/file/d/1FQPLKfILW-rEoXvLOEZG5Zm4YF6fXcKh/view?usp=sharing) | [Decoder](https://drive.google.com/file/d/1kNa6PtS_dVK-IqLaIicn9IqbGNQJ8zmJ/view?usp=sharing)| 0.001
HPD| [Pose Endcoder](https://drive.google.com/file/d/19Vhbhlw6hhIcNoECw57ze2cRDmhrbSMB/view?usp=sharing) | [Decoder](https://drive.google.com/file/d/1EO3XYN7dEO1QYF1N6NiUgGkn7ss7mslm/view?usp=sharing)| 0.001



## Setup

This section shows how to setup this repository including installing requirements (including the software versions used), setting up the files to run this repository, etc. 

### Requirements

- Python X.X
- PyTorch X.X
- CUDA X.X

### Dependencies

Install all the Python dependencies using pip:

~~~
pip install -r requirements.txt
~~~

Install Ubuntu libraries with: 

~~~
apt-get install some-software
~~~

## Useful Features and/or Processed Data for Future Projects

This section mentions potential useful processed data or features that might be useful for future projects (e.g. bounding box proposals).

- `PATH/TO/FEATURES`: a description of what it is.

For example,

- `data/processed/proposals`, you could find the bounding box proposals used. 

## Training

This section provides instructions to training this model from scratch. 

For example,
~~~
python main.py --config-file configs/faster_rcnn_res101.yaml
~~~

## Evaluate

This section shows how to evaluate a trained model on a dataset to reproduce the benchmark performance. 

For example,
~~~
python main.py --config-file configs/faster_rcnn_res101.yaml --inference --resume $CHECKPOINT
~~~

## Contact and Reference

- For more information, please contact REPO-OWNER@cs.ubc.ca

~~~
@inproceedings{latex-citation-name,
    title={THE TITLE OF YOUR REPORT/PAPER},
    author={LAST NAME, FIRST NAME},
    booktitle={REPORT TYPE/CONFERENCE TYPE},
    year={20XX}
}
~~~
