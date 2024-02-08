
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
├── TRAIN                   # contain training files
│   ├── T1
│   │   ├── kspace
│   │   │   ├── train_1.mat         
│   │   │   ├── train_2.mat 
│   │   │   ├── ...         
│   │   │   └── train_N.mat 
│   │   └── ...
│   │   
│   ├── T2
│   │   ├── kspace
│   │   │   ├── train_1.mat          
│   │   │   ├── train_2.mat 
│   │   │   ├── ...         
│   │   │   └── train_N.mat 
│   │   └── ...
│   │   
│   ├── FLAIR
│   │   ├── kspace
│   │   │   ├── train_1.mat          
│   │   │   ├── train_2.mat 
│   │   │   ├── ...         
│   │   │   └── train_N.mat 
│   │   └── ...
│   └── ...
│
├── VALI                    # contain validation files
│   ├── T1
│   │   ├── kspace
│   │   │   ├── vali_1.mat          
│   │   │   ├── vali_2.mat 
│   │   │   ├── ...         
│   │   │   └── vali_M.mat 
│   │   └── ...
│   │   
│   ├── T2
│   │   ├── kspace
│   │   │   ├── vali_1.mat          
│   │   │   ├── vali_2.mat 
│   │   │   ├── ...         
│   │   │   └── vali_M.mat 
│   │   └── ...
│   │   
│   ├── FLAIR
│   │   ├── kspace
│   │   │   ├── vali_1.mat          
│   │   │   ├── vali_2.mat 
│   │   │   ├── ...         
│   │   │   └── vali_M.mat 
│   │   └── ...
│   └── ...
│
├── TEST                    # contain test files
│   ├── T1
│   │   ├── kspace
│   │   │   ├── test_1.mat          
│   │   │   ├── test_2.mat 
│   │   │   ├── ...         
│   │   │   └── test_K.mat 
│   │   └── ...
│   │   
│   ├── T2
│   │   ├── kspace
│   │   │   ├── test_1.mat          
│   │   │   ├── test_2.mat 
│   │   │   ├── ...         
│   │   │   └── test_K.mat 
│   │   └── ...
│   │   
│   ├── FLAIR
│   │   ├── kspace
│   │   │   ├── test_1.mat          
│   │   │   ├── test_2.mat 
│   │   │   ├── ...         
│   │   │   └── test_K.mat 
│   │   └── ...
│   └── ...
│            
└── ...
```
## Training

To train the model(s) in the paper, run this command:

```train
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
>
## Evaluation

To evaluate my model on MRI dataset, e.g., IXI, BraTS, fastMRI, run:

```eval
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI --train test
```


