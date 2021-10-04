import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A


class cd2014(Dataset):

    def __init__(self, root,split_flag,
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 ):
        super(cd2014, self).__init__()
        self.flag = split_flag
        if self.flag == 'train':
            self.img_txt_path = os.path.join(root,'train.txt')
        elif self.flag == 'val':
            self.img_txt_path = os.path.join(root,'val.txt')

        self.img_path = root
        self.label_path = root
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.paths = {
            'GT':[path[2] for k, path in self.img_label_path_pairs.items()],
            'query': [path[1] for k, path in self.img_label_path_pairs.items()],
            'ref': [path[0] for k, path in self.img_label_path_pairs.items()]
        }

        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag == 'train':
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]
                # print extract_name
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, image2_name])

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]
                # print extract_name
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, image2_name])

        if self.flag == 'test':

            for idx, did in enumerate(open(self.img_txt_path)):
                image1_name, image2_name = did.strip("\n").split(' ')
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, None, image2_name])

        return img_label_pair_list

    def __getitem__(self, index):

        fn_t0 = self.paths['ref'][index]
        fn_t1 = self.paths['query'][index]
        fn_mask = self.paths['GT'][index]
        print(fn_t0)
        print(fn_t1)
        print(fn_mask)
        fn_mask_npy = self.paths['GT'][index].replace('gt_binary','GT_npy')
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
            mask_binary = self.change_transform(mask_r_)

        return {'source_image': img_t0_r_,
                'target_image': img_t1_r_,
                'source_change': mask_r_.squeeze(),
                'target_change': mask_r_.squeeze(),
                'source_image_size': (h_r,w_r,3)
                }

    def __len__(self):
        return len(self.paths['GT'])


if __name__ == '__main__':
    dataset = cd2014(root='/media/rit/GLU-CHANGE-SSD500/dataset/CD2014',
                     split_flag='train')
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
