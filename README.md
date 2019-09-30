# YOUR_PROJECT_NAME

Give a few sentences to describe this project. 

## Benchmarking

This section stores the trained model(s) for this project and its benchmark performance. 

### Object Detection

source  | backbone | model | bs | lr  | lr_decay | mAP@0.5 | mAP@0.50:0.95
--------|--------|--------|:------:|:------:|:-------:|:------:|:------:
[LINK_TO_TRAINED_MODEL](URL-TO-TRAINED-MODEL) | Res-101 | faster r-cnn | 6 | 5e-3 | 70k,90k | 24.8 | 12.8

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

## Useful Features/Processed Data

- At `data/preprocessed/features`, you could find the features used in this project. 
- At `data/preprocessed/proposals`, you could find the bounding box proposals used. 

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

- For more information, please contact xxxx@cs.ubc.ca

~~~
@inproceedings{latex-citation-name,
    title={THE TITLE OF YOUR REPORT/TITLE},
    author={LAST NAME, FIRST NAME},
    booktitle={REPORT TYPE/CONFERENCE TYPE},
    year={20XX}
}
~~~
