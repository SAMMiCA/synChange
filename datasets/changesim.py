import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A
from natsort import natsorted
class SegHelper:
    def __init__(self,opt=None,idx2color_path='./datasets/idx2color.txt',num_class=5):
        self.opt = opt
        self.num_classes = num_class
        self.idx2color_path = idx2color_path
        f = open(self.idx2color_path, 'r')
        self.idx2color = {k:[] for k in range(self.num_classes)}
        for j in range(num_class):
            line = f.readline()
            line = line.strip(' \n').strip('[').strip(']').strip(' ').split()
            line = [int(l) for l in line if l.isdigit()]
            self.idx2color[j] = line # color in rgb order

        self.color2idx = {tuple(v):k for k,v in self.idx2color.items()}


    def unique(self,array):
        uniq, index = np.unique(array, return_index=True, axis=0)
        return uniq[index.argsort()]

    def extract_color_from_seg(self,img_seg):
        colors = img_seg.reshape(-1, img_seg.shape[-1]) # (H*W,3) # color channel in rgb order
        unique_colors = self.unique(colors) # (num_class_in_img,3)
        return unique_colors

    def extract_class_from_seg(self,img_seg):
        unique_colors = self.extract_color_from_seg(img_seg) # (num_class_in_img,3) # color channel in rgb order
        classes_idx = [self.color2idx[tuple(color.tolist())]for color in unique_colors]
        classes_str = [self.idx2name[idx] for idx in classes_idx]
        return classes_idx, classes_str

    def colormap2classmap(self,seg_array):
        seg_array_flattened = torch.LongTensor(seg_array.reshape(-1,3)).cuda()
        seg_map_class_flattened = torch.zeros((seg_array.shape[0],seg_array.shape[1],1)).view(-1,1).cuda()
        for color, cls in self.color2idx.items():
            matching_indices = (seg_array_flattened == torch.LongTensor(color).cuda())
            matching_indices = (matching_indices.sum(dim=1)==3)
            seg_map_class_flattened[matching_indices] = cls
        seg_map_class = seg_map_class_flattened.view(seg_array.shape[0],seg_array.shape[1],1)
        return seg_map_class

    def classmap2colormap(self,seg_map_class):
        seg_map_class_flattened = seg_map_class.view(-1,1)
        seg_map_color_flattened = torch.zeros(seg_map_class.shape[0]*seg_map_class.shape[1],3).cuda().long()
        for cls, color in self.idx2color.items():
            matching_indices = (seg_map_class_flattened == torch.LongTensor([cls]).cuda())
            seg_map_color_flattened[matching_indices.view(-1)] = torch.LongTensor(color).cuda()
        seg_map_color_flattened = seg_map_color_flattened.view(seg_map_class.shape[0],seg_map_class.shape[1],3)
        return seg_map_color_flattened


class changesim_eval(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 multi_class = False,
                 mapname='*',
                 seqname='Seq_0',
                 img_size=(640, 480)
                 ):
          # 파일 생성일

        super(changesim_eval, self).__init__()

        self.paths = {'GT':natsorted(glob(pjoin(root,mapname,seqname,'change_segmentation','*.png')))}
        self.paths['GT'] = [fn_mask for fn_mask in self.paths['GT'] if os.path.isfile(fn_mask)]
        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('change_segmentation','rgb')
            query_paths.append(query_path)
            ref_path = query_path.replace('rgb','t0/rgb')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform
        self.seghelper = SegHelper()
        self.colors = self.seghelper.idx2color
        self.num_class = 5 if multi_class else 1
        self.seqname = seqname
        self.mapname = mapname
        self.img_size = img_size
    def __getitem__(self, index):
        fn_t0 = self.paths['ref'][index]
        if 'fire' in self.seqname: index= index+5
         #if 'dark' in self.seqname: index= index+ 10
        fn_mask = self.paths['GT'][index]
        fn_mask_npy = self.paths['GT'][index].replace('change_segmentation','GT_npy')
        fn_mask_npy = fn_mask_npy.replace('png','npy')


        fn_t1 = self.paths['query'][index]

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            import pdb; pdb.set_trace()
            # exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            import pdb; pdb.set_trace()
            # exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            import pdb; pdb.set_trace()
            # exit(-1)
        if os.path.isfile(fn_mask_npy) == False:
            print('Error: File Not Found: ' + fn_mask_npy)
            mask_color = cv2.imread(fn_mask, 1)
            if not os.path.isdir(os.path.dirname(fn_mask_npy)):
                os.makedirs(os.path.dirname(fn_mask_npy),exist_ok=True)
            mask_color = np.asarray(mask_color).astype('uint8')
            mask = self.colormap2classmap(mask_color[:,:,[2,1,0]]) # (h,w,1)
            np.save(fn_mask_npy,mask.numpy())
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        # w, h, c = img_t0.shape
        # w_r = 360 # int(256 * max(w / 256, 1))
        # h_r = 480 # int(256 * max(h / 256, 1))
        h_r, w_r = self.img_size

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')
        mask_r_ = np.asarray(mask_r[:,:,None]).astype('uint8')
        mask_reordered = np.zeros_like(mask_r_)
        mask_reordered[mask_r_==1]=2
        mask_reordered[mask_r_==2]=1
        mask_reordered[mask_r_==4]=3
        mask_reordered[mask_r_==3]=4
        mask_r_ = mask_reordered

        # mask_binary = (mask_r_>1).astype('uint8')

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

        if self.num_class == 1: # multi-class classification -> binary classification
            mask_r_ = mask_r_>0
            mask_r_ = mask_r_.astype('uint8')
        if self.change_transform is not None:
            mask_r_ = self.change_transform(mask_r_)



        return {'source_image': img_t0_r_,
                'target_image': img_t1_r_,
                'source_change': mask_r_.squeeze().int(),
                'target_change': mask_r_.squeeze().int(),
                # 'source_image_size': (h_r,w_r,3),
                'flow_map': torch.zeros(2, img_t1_r_.shape[1], img_t1_r_.shape[2]),
                'correspondence_mask': torch.ones_like(mask_r_.squeeze()).numpy().astype(np.bool),
                'use_flow': torch.zeros(1),
                'disable_flow': torch.zeros(1),

                }

    def __len__(self):
        if 'fire' in self.seqname:
            return len(self.paths['GT']) - 10
        # if 'dark' in self.seqname and 'Room_0' in self.mapname:
        #     return len(self.paths['GT']) - 10
        return len(self.paths['GT'])

    def unique(self,array):
            uniq, index = np.unique(array, return_index=True, axis=0)
            return uniq[index.argsort()]

    def colormap2classmap(self, seg_array):
        seg_array_flattened = torch.LongTensor(seg_array.reshape(-1,3))
        seg_map_class_flattened = torch.zeros((seg_array.shape[0],seg_array.shape[1],1)).view(-1,1)
        for cls,color in self.colors.items():
            matching_indices = (seg_array_flattened == torch.LongTensor(color))
            matching_indices = (matching_indices.sum(dim=1)==3)
            seg_map_class_flattened[matching_indices] = cls
        seg_map_class = seg_map_class_flattened.view(seg_array.shape[0],seg_array.shape[1],1)
        return seg_map_class

if __name__ == '__main__':
    dataset = changesim_eval(root='/media/rit/GLU-CHANGE-SSD500/dataset/ChangeSim',multi_class = True)
    import matplotlib.pyplot as plt
    i=0
    while(1):
        sample = dataset[i]
        # plt.subplot(311)
        # plt.imshow(50*sample['target_change'][:,:,0],vmin=0,vmax=255)
        # plt.subplot(312)
        # plt.imshow(sample['source_image'])
        # plt.subplot(313)
        # plt.imshow(sample['target_image'])
        #plt.show()
        # plt.close()
        print(i)
        i+=1
