# YOUR_PROJECT_NAME

If you have trouble archiving the project please contact Alex Fan (fan@cs.ubc.ca). This is a template repository, please modify it according to your project. 

## Introduction
Give a few sentences to tell us about your project. 

## Benchmarking

This section stores the trained model(s) for this project and its benchmark performance. 

### Object Detection

source  | backbone | model | bs | lr  | lr_decay | mAP@0.5 | mAP@0.50:0.95
--------|--------|--------|:------:|:------:|:-------:|:------:|:------:
[LINK_TO_TRAINED_MODEL](URL-TO-TRAINED-MODEL) | Res-101 | faster r-cnn | 6 | 5e-3 | 70k,90k | 24.8 | 12.8


## Directory structure

The overall directory structure of your new project looks like this (feel free to add more folders or files): 

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- Processed data such as the extracted features, bounding box proposals, etc.
│   └── raw            <- The original dataset obtained. 
│
├── models             <- Trained model weights, e.g. created by `torch.save(model.state_dict(), ..)`.
│
├── notebooks          <- Jupyter notebooks for cool demos. 
│
├── reports            <- Detailed project reports
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Data processing scripts, dataset class definitions, dataloader definitions. 
    │
    └── models         <- Related to model training and evaluation. 
```

## Setup

This section shows how to setup this repository including installing requirements (including the software versions used), setting up the files to run this repository, etc. 

### Requirements

- Python 3.6
- PyTorch 1.0
- CUDA 8.0

### Dependencies

Install all the python dependencies using pip:
~~~
pip install -r requirements.txt
~~~

Install Ubuntu libraries with: 
~~~
apt-get install some-software
~~~

## Useful Features and/or Processed Data for Future Projects

- At `data/processed/features`, you could find the features used in this project. 
- At `data/processed/proposals`, you could find the bounding box proposals used. 

## Training

This section provides instructions to training this model from scratch. 

Train object detection model:
~~~
python main.py --config-file configs/faster_rcnn_res101.yaml
~~~

## Evaluate

This section shows how to evaluate a trained model on a dataset to reproduce the benchmark performance. 

Evaluate object detection model:

~~~
python main.py --config-file configs/faster_rcnn_res101.yaml --inference --resume $CHECKPOINT
~~~

## Contact and Reference

- For more information, please contact REPO-OWNER@cs.ubc.ca

~~~
@inproceedings{latex-citation-name,
    title={THE TITLE OF YOUR REPORT/TITLE},
    author={LAST NAME, FIRST NAME},
    booktitle={REPORT TYPE/CONFERENCE TYPE},
    year={20XX}
}
~~~
