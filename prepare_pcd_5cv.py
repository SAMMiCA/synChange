import torch
import os
import torchvision
import cv2
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import save_image
from PIL import Image
for set_name in ['set0','set1','set2','set3','set4']:
    for split in ['train','test']:
        f = open(os.path.join('../dataset/test_datasets/pcd_5cv',set_name,'{}.txt'.format(split)))
        paths = f.readlines()
        annotation = open(os.path.join('../dataset/test_datasets/pcd_5cv',set_name,split,'tsunami_or_gsv.txt'),mode='w')
        print('# samples in {}: {}'.format(set_name+'/'+split,len(paths)))
        tsunami_imgs = []
        gsv_imgs = []


        for idx,(t0_t1_mask) in enumerate(paths):
            t0_path, _, _ = t0_t1_mask.split(' ')
            t0_path = os.path.join('dataset/test_datasets/pcd_5cv',set_name,split,t0_path)
            img_t0 = cv2.imread(t0_path, 1)
            img_t0 = cv2.resize(img_t0, (128,128))
            if idx % 120 < 60:
                tsunami_imgs.append(torch.LongTensor(img_t0).permute(2,0,1))
                annotation.write(t0_path.split('/')[-1].replace('jpg','png')+' '+'0\n')
            else:
                gsv_imgs.append(torch.LongTensor(img_t0).permute(2,0,1))
                annotation.write(t0_path.split('/')[-1].replace('jpg','png')+' '+'1\n')
        annotation.close()
        # grid_tsunami = torchvision.utils.make_grid(tsunami_imgs,nrow=32,padding=0)
        # cv2.imwrite('{}_{}_tsunami.jpg'.format(set_name,split),grid_tsunami.permute(1,2,0).numpy().astype(np.uint8))
        # grid_gsv = torchvision.utils.make_grid(gsv_imgs,nrow=32,padding=1)
        # cv2.imwrite('{}_{}_gsv.jpg'.format(set_name,split),grid_gsv.permute(1,2,0).numpy().astype(np.uint8))
