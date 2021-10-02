import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A
import natsort
def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','png']])


class desk_demo(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 ):
        super(desk_demo, self).__init__()
        self.paths = {'query':natsort.natsorted(glob(pjoin(root,'t1','*.png')))}
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
    def __getitem__(self, index):

        fn_t1 = self.paths['query'][index]
        fn_t0 = os.path.dirname(self.paths['query'][index].replace('t1','t0'))
        fn_t0 = os.path.join(fn_t0,'0.png')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)


        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        w, h, c = img_t0.shape
        w_r = 480#int(256 * max(w / 256, 1))
        h_r = 640#int(256 * max(h / 256, 1))

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')

        if self.source_image_transform is not None:
            if isinstance(self.source_image_transform,transforms.Compose):
                img_t1_r_ = self.source_image_transform(img_t1_r_)
            elif isinstance(self.source_image_transform,A.Compose):
                img_t1_r_ = self.source_image_transform(image=img_t1_r_)['image']
            else: raise ValueError
        if self.target_image_transform is not None:
            if isinstance(self.target_image_transform,transforms.Compose):
                img_t0_r_ = self.target_image_transform(img_t0_r_)
            elif isinstance(self.target_image_transform,A.Compose):
                img_t0_r_ = self.target_image_transform(image=img_t0_r_)['image']
            else: raise ValueError

        return {'source_image': img_t0_r_,
                'target_image': img_t1_r_,
                'source_image_size': (h_r,w_r,3)
                }

    def __len__(self):
        return len(self.paths['query'])



if __name__ == '__main__':
    dataset = desk_demo(root='/media/rit/GLU-CHANGE-SSD500/dataset/desk')
    import matplotlib.pyplot as plt
    i=0
    while(1):
        sample = dataset[i]
        # import pdb; pdb.set_trace()
        plt.subplot(211)
        plt.imshow(sample['source_image'])
        plt.subplot(212)
        plt.imshow(sample['target_image'])
        plt.show()
        plt.close()
        print(i)
        i+=1
