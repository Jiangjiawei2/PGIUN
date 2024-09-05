
# PGIUN:Physics-Guided Implicit Unrolling Network for Accelerated MRI

This repository is the official implementation of [PGIUN:Physics-Guided Implicit Unrolling Network for Accelerated MRI](https://ieeexplore.ieee.org/abstract/document/10584139), accepted by TCI. If you have any questions, please feel free to contact me："jjw@zjut.edu.cn" 

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
> Configure ```data_dir``` and ```root_path``` in the ```config.yaml``` folder, and configure the ```config.yaml``` path in ```option.py```.
## Training

To train the model(s) in the paper, run this command:

```train
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI
```
where  
```--model``` provides the model name for the current run.  
```--mask``` provides the mask used in the current run.  
```--acceleration``` defines the acceleration ratio.  
```--data_name``` provides the data name of the current run.  
Other hyperparameters can be adjusted in the code as well.  

## Evaluation

To evaluate the model on MRI dataset, e.g., IXI, BraTS, fastMRI, run:

```eval
python mc_rec_main.py --model pgiun --batch_size 1 --n_epochs 100 --mask random --gpuid 0 --modal T2 --acceleration 4 --data_name IXI --train test
```

## Citation
If you find it helpful, please cite our literature:
```
@article{jiang2024pgiun,
  title={PGIUN: Physics-Guided Implicit Unrolling Network for Accelerated MRI},
  author={Jiang, Jiawei and He, Zihan and Quan, Yueqian and Wu, Jie and Zheng, Jianwei},
  journal={IEEE Transactions on Computational Imaging},
  year={2024},
  publisher={IEEE}
}
```

