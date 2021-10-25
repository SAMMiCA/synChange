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

    def __init__(self, root, split='train',
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 img_size=(640, 480)
                 ):
        super(cd2014, self).__init__()
        self.split = split
        if self.split == 'train':
            self.img_txt_path = os.path.join(root,'train.txt')
        else:
            self.img_txt_path = os.path.join(root,'val.txt')

        self.img_path = root
        self.label_path = root
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.paths = {
            'GT':[path[2] for k, path in self.img_label_path_pairs.items() if 'PTZ' not in path[2]],
            'query': [path[1] for k, path in self.img_label_path_pairs.items() if 'PTZ' not in path[1]],
            'ref': [path[0] for k, path in self.img_label_path_pairs.items() if 'PTZ' not in path[0]],
            'roi': [os.path.join(os.path.dirname(os.path.dirname(path[0])),'ROI.bmp') for k, path in self.img_label_path_pairs.items()
                    if 'PTZ' not in path[0]]
        }

        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform
        self.img_size = img_size

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.split == 'train':
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

        if self.split == 'test':
            # self.label_ext = '.png'
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

        # if self.split == 'test':
        #
        #     for idx, did in enumerate(open(self.img_txt_path)):
        #         image1_name, image2_name = did.strip("\n").split(' ')
        #         img1_file = os.path.join(self.img_path, image1_name)
        #         img2_file = os.path.join(self.img_path, image2_name)
        #         img_label_pair_list.setdefault(idx, [img1_file, img2_file, None, image2_name])

        return img_label_pair_list

    def __getitem__(self, index):

        fn_t0 = self.paths['ref'][index]
        fn_t1 = self.paths['query'][index]
        fn_mask = self.paths['GT'][index]
        fn_roi = self.paths['roi'][index]
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
        if os.path.isfile(fn_roi) == False:
            print('Error: File Not Found: ' + fn_roi)
        else:
            roi = cv2.imread(fn_roi)  # HWC, dtype=uint8, valid region =255, invalid region = 0
            roi = roi[:,:,0]/255 # HW, valid=1, invalid=0
            roi_r_ = cv2.resize(roi,self.img_size,interpolation=cv2.INTER_NEAREST)
            roi_r_ = np.asarray(roi_r_).astype('uint8')

        img_t0 = cv2.imread(fn_t0, 1) # default color order for cv2 is bgr
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        h_r, w_r = self.img_size
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
            # roi_r_ = self.change_transform(roi_r_)

        return {'source_image': img_t0_r_,
                'target_image': img_t1_r_,
                'source_change': mask_r_.squeeze().int(),
                'target_change': mask_r_.squeeze().int(),
                'flow_map': torch.zeros(2, img_t1_r_.shape[1], img_t1_r_.shape[2]),
                'correspondence_mask': roi_r_.astype(np.bool),
                'use_flow': torch.zeros(1),  # if true, use gt flow map for training with this sample
                'disable_flow': torch.zeros(1), # if true, disable warping when training with this sample
                }

    def __len__(self):
        return len(self.paths['GT'])


if __name__ == '__main__':
    dataset = cd2014(root='/media/rit/GLU-CHANGE-SSD500/dataset/CD2014',
                     split='train')
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
