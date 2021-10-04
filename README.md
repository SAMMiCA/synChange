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
sudo scp -r rit@143.248.151.68:/home/rit/E2EChangeDet/dataset/train_datasets .
```

# Prepare Test set
```bash
sudo scp -r rit@143.248.151.68:/home/rit/E2EChangeDet/dataset/test_datasets .
```

