# YOUR_PROJECT_NAME

Give a few sentences to describe this project. 

## Benchmarking

### Object Detection

source  | backbone | model | bs | lr  | lr_decay | mAP@0.5 | mAP@0.50:0.95
--------|--------|--------|:------:|:------:|:-------:|:------:|:------:
[this repo](https://drive.google.com/open?id=1THLvK8q2VRx6K3G7BGo0FCe-D0EWP9o1) | Res-101 | faster r-cnn | 6 | 5e-3 | 70k,90k | 24.8 | 12.8

## Setup

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

## Training

Train object detection model:
~~~
python main.py --config-file configs/faster_rcnn_res101.yaml
~~~

## Evaluate

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
