import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A

class ai2thor(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 ):
        super(ai2thor, self).__init__()
        self.paths = {'GT':glob(pjoin(root,'mask','*','*.png'))}
        if not os.path.isdir(pjoin(root,'GT_npy')):
            os.mkdir(pjoin(root,'GT_npy'))
        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('mask','t1')
            query_paths.append(query_path)
            ref_path = query_path.replace('t1','t0')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform
    def __getitem__(self, index):

        fn_t0 = self.paths['query'][index]
        fn_t1 = self.paths['ref'][index]
        idx = int(fn_t1.split('/')[-1].split('.')[0])
        if idx>2:
            fn_t1 = '/'.join(fn_t1.split('/')[:-1])
            fn_t1 = os.path.join(fn_t1,'{:03d}.png'.format(idx-2))

        fn_mask = self.paths['GT'][index]
        fn_mask_npy = self.paths['GT'][index].replace('mask','GT_npy')
        fn_mask_npy = fn_mask_npy.replace('png','npy')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            exit(-1)
        if os.path.isfile(fn_mask_npy) == False:
            print('Error: File Not Found: ' + fn_mask_npy)
            mask_color = cv2.imread(fn_mask, 1)
            if not os.path.isdir(os.path.dirname(fn_mask_npy)):
                os.mkdir(os.path.dirname(fn_mask_npy))
            # import pdb; pdb.set_trace()
            mask = np.asarray(mask_color).astype('uint8')[:,:,0]>0
            mask = mask[...,None]
            np.save(fn_mask_npy,mask.astype('uint8'))
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        w, h, c = img_t0.shape
        w_r = int(256 * max(w / 256, 1))
        h_r = int(256 * max(h / 256, 1))

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))

        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')
        mask_r_ = np.asarray(mask_r[:,:,None]).astype('uint8')

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

        if self.change_transform is not None:
            mask_r_ = self.change_transform(mask_r_)

        return {'source_image': img_t1_r_,
                'target_image': img_t0_r_,
                'source_change': mask_r_.squeeze(),
                'target_change': mask_r_.squeeze(),
                'source_image_size': (h_r,w_r,3)
                }

    def __len__(self):
        return len(self.paths['GT'])



if __name__ == '__main__':
    dataset = ai2thor(root='/media/rit/GLU-CHANGE-SSD500/dataset/ai2thor')
    import matplotlib.pyplot as plt
    i=0
    while(1):
        sample = dataset[i]
        # plt.subplot(311)
        # plt.imshow(sample['source_image'])
        # plt.subplot(312)
        # plt.imshow(sample['target_image'])
        # plt.subplot(313)
        # print(sample['target_change'].max())
        # plt.imshow(sample['target_change'],vmax=1)
        # plt.show()
        # plt.close()
        print(i)
        i+=1
