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





# Dataset Generation
```bash
python save_change_training_dataset_to_disk.py --save_dir result --plot True
```
