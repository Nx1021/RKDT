# <center> RGB-Based Set Prediction Transformer of 6D Pose Estimation for Robotic Grasping Application

Paper url: *TODO*

## Framework
![framework](image\method.png)

## Requirements
* h5py==3.7.0
* matplotlib==3.5.2
* MyLib==0.0.1
* numpy==1.22.3
* open3d==0.17.0
* opencv_python==4.6.0.66
* pandas==1.4.3
* Pillow==9.1.1
* Pillow==10.0.0
* python_Levenshtein==0.21.1
* scikit_learn==1.2.2
* scipy==1.8.1
* sko==0.5.7
* tensorboard==2.10.0
* torch==1.12.1+cu113
* torchvision==0.13.1+cu113
* tqdm==4.65.0
* ultralytics==8.0.119

## Installation
This code has been tested on :
* Ubuntu22.04 with Cuda 12.5, Python3.9 and Pytorch 1.12.1.
* Win10 with Cuda 12.4, Python3.9 and Pytorch 1.12.1.

Get the code
```bash
git clone https://github.com/Nx1021/RKDT.git
```
Create new Conda environment.
```
conda create -n rkdt python=3.9
conda activate RKDT
```
Install torch
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
Others
```
pip install -r requirements.txt
```

## Usage
### Checkpoint
TODO
### Dataset
TODO
### Train
TODO
### Test
TODO

## Results
| Object    | ADD(-S) | 2D-proj | 
| :----     | :----:  | :----:  | 
| ape       | 91.05   | 99.81   |
| bench.    | 97.19   | 96.61   |
| cam       | 88.39   | 95.67   |
| can       | 98.23   | 99.61   |
| cat       | 88.22   | 99.90   |
| driller   | 97.82   | 98.41   |
| duck      | 80.09   | 99.81   |
| eggbox    | 99.25   | 99.06   |
| glue      | 96.62   | 99.71   |
| holep.    | 79.35   | 99.52   |
| iron      | 98.47   | 98.26   |
| lamp      | 98.46   | 97.79   |
| phone     | 94.79   | 99.15   |
| average   | 92.92   | 98.72   |
## Citation
TODO