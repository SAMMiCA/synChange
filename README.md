# GLU-ChangeNet

# Installation

0. Prerequisites

- Python 3.7 or later
- Pytorch 1.8.1 or later
- cuda 11.1 or later 

1. Install requirements

```bash
pip install -r requirements.txt
```
2. Install cupy

```bash
pip install cupy-cuda111 --no-cache-dir
```

# Prepare Pre-trained Model
```bash
cd GLU-ChangeNet-Pytorch
sudo scp -r rit@143.248.151.68:/home/rit/E2EChangeDet/GLU-ChangeNet-Pytorch/pre_trained_models .
```

# Prepare Train set
```bash
python save_change_training_dataset_to_disk.py --save_dir result --plot True
```
or just download one
```bash
cd ..
mkdir dataset
cd dataset
mkdir dataset/train_datasets
cd train_datasets
sudo scp -r rit@143.248.151.68:/home/rit/E2EChangeDet/dataset/train_datasets/synthetic .
cd synthetic/flow
ln -s static new
ln -s ststic replaced
ln -s static missing
```

# Prepare Test set
```bash
cd ../../.. 
# In 'dataset' folder
sudo scp -r rit@143.248.151.68:/home/rit/E2EChangeDet/dataset/test_datasets .
```

# Init Training for Single-class Change Detection
```bash
cd ../GLU-ChangeNet-Pytorch
python train_GLUChangeNet.py \
--training_data_dir ../dataset/train_datasets/synthetic_dataset \
--evaluation_data_dir ../dataset/test_datasets \
--pretrained pre_trained_models/GLUNet_DPED_CityScape_ADE.pth \
--n_threads 4 --plot_interval 1 --split_ratio 0.90 --split2_ratio 0.5 \
--trainset_list synthetic vl_cmu_cd \
--testset_list vl_cmu_cd \
--lr 0.0002 --n_epoch 25 \
--name_exp joint_synthetic_vl-cmu-cd


```
# Init Training for Multi-class Change Detection
```bash
cd ../GLU-ChangeNet-Pytorch
--training_data_dir ../dataset/train_datasets/synthetic_dataset \
--evaluation_data_dir ../dataset/test_datasets \
--pretrained pre_trained_models/GLUNet_DPED_CityScape_ADE.pth \
--n_threads 4 --plot_interval 1 --split_ratio 0.90 --split2_ratio 0.5 \
--trainset_list synthetic \
--testset_list changesim \
--lr 0.0002 --n_epoch 25 \
--name_exp train_synthetic_test_changesim_multiclass \
--multi_class
```
