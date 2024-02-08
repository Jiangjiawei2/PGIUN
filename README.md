
# PGIUN:Physics-Guided Implicit Unrolling Network for Accelerated MRI

This repository is the official implementation of [PGIUN:Physics-Guided Implicit Unrolling Network for Accelerated MRI], currently submitting to TCI. If you have any questions, please feel free to contact me："jjw@zjut.edu.cn" 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
```
einops==0.4.1
ipdb==0.13.9
layers==0.1.5
matplotlib==3.5.2
numpy==1.20.3
PyYAML==6.0.1
scikit_image==0.19.3
scipy==1.7.3
setuptools==59.5.0
SimpleITK==2.3.1
thop==0.1.1.post2209072238
timm==0.5.4
torch==1.13.1+cu117
torchvision==0.14.1+cu117
tqdm==4.64.1
```

## Dataset Setup
```
Data
│   ├── T1
│   │   ├── train
│   │   │   ├── train_1.npy         
│   │   │   ├── train_2.npy 
│   │   │   ├── ...         
│   │   │   └── train_N.npy
│   │   └── valid
│   │   │   ├── valid_1.npy         
│   │   │   ├── valid_2.npy 
│   │   │   ├── ...         
│   │   │   └── valid_N.npy
│   │   └── test
│   │   │   ├── test_1.npy         
│   │   │   ├── test_2.npy 
│   │   │   ├── ...         
│   │   │   └── test_N.npy
│   │   
│   ├── T2
│   │   ├── train
│   │   │   ├── train_1.npy         
│   │   │   ├── train_2.npy 
│   │   │   ├── ...         
│   │   │   └── train_N.npy
│   │   └── valid
│   │   │   ├── valid_1.npy         
│   │   │   ├── valid_2.npy 
│   │   │   ├── ...         
│   │   │   └── valid_N.npy
│   │   └── test
│   │   │   ├── test_1.npy         
│   │   │   ├── test_2.npy 
│   │   │   ├── ...         
│   │   │   └── test_N.npy
│   │   
│   │   └── ...
│   └── ...
│            
└── ...
```
> Put the ```data path``` to data _ dir in config.yaml.
## Training

To train the model(s) in the paper, run this command:

```train
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI
```
where
```--model``` provides the experiment name for the current run.  
```--mask``` provides the mask used in the current experiment.  
```--acceleration``` defines acceleration ratio.  
```--data_name``` provides the data name of the current run.  
Other hyperparameters can be adjusted in the code as well.  

## Evaluation

To evaluate the model on MRI dataset, e.g., IXI, BraTS, fastMRI, run:

```eval
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI --train test
```


