# VPR

All codes were modified based on the official codebase of each VPR module.

[Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition](https://github.com/QVPR/Patch-NetVLAD)

[STA-VPR: Spatio-temporal Alignment for Visual Place Recognition](https://github.com/Lu-Feng/STA-VPR)

[A Hierarchical Dual Model of Environment- and Place-Specific Utility for Visual Place Recognition](https://github.com/Nik-V9/HEAPUtil)

## Patch-NetVLAD & NetVLAD

Run the following command for feature extraction: 

`python descriptor_extract.py --query_data dir /your/query_data/dir --ref_data_dir /your/ref_data/dir --descriptors_path /your/output/descriptors/dir --dataset /your/dataset/name`

Run the following command for matching: 

`python descriptor_match.py --query_data_dir /your/query_data/dir --ref_data_dir /your/ref_data/dir --query_features_dir /your/query_descriptors/dir --ref_features_dir /your/ref_descriptors/dir --result_save_folder /output/predictions/dir --dataset /your/dataset/name`

## STA-VPR

Modify each argument in 'STAVPRconfig.yaml' for your purpose and run the following command: `python STAVPRdemo.py`

## HEAPUtil

Run 'gen_data_json.py' to generate the json file that includes the information of the dataset.
Run each command in 'HEAPUtil_Demo.ipynb'.  
