import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A

# Train and Test splits
TRAIN_VIDEOS = '1 2 3 4 5 8 10 11 13 14 15 16 17 18 19 20 21 22 26 29 30 31 33 35 37 40 41 42 43 44 46 49 51 52 53 54 55 57 59 62 63 65 67 68 70 71 72 73 74 75 78 79 80 83 84 86 87 88 89 90 91 96 98 99 101 102 103 104 105 108 109 110 111 114 115 116 118 121 122 123 124 126 127 128 130 131 133 136 137 138 140 141 143 146 147 148 149 151 '
TEST_VIDEOS = '0 6 7 9 12 23 24 25 27 28 32 34 36 38 39 45 47 48 50 56 58 60 61 64 66 69 76 77 81 82 85 92 93 94 95 97 100 106 107 112 113 117 119 120 125 129 132 134 135 139 142 144 145 150 '

class vl_cmu_cd_eval(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 split='train',
                 img_size = (640,480)
                 ):
        super(vl_cmu_cd_eval, self).__init__()

        self.paths = {'GT':glob(pjoin(root,'*','GT','*.png'))}
        video_idxs = TRAIN_VIDEOS if split == 'train' else TEST_VIDEOS
        video_idxs = video_idxs.strip().split(' ')
        self.paths['GT'] = [path for path in self.paths['GT'] if str(int(path.split('/')[-3])) in video_idxs]
        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('GT','RGB')
            query_path = query_path.replace('gt','1_')
            query_paths.append(query_path)
            ref_path = query_path.replace('1_','2_')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform
        self.img_size = img_size
        self.colors = np.array([[  0,   0,   0],
                                [255, 255, 255],
                                 [201, 174, 255],
                                 [164,  73, 163],
                                 [ 36,  28, 237],
                                 [ 29, 230, 181],
                                 [ 76, 177,  34],
                                 [ 39, 127, 255],
                                 [204,  72,  63],
                                 [ 21,   0, 136],
                                 [232, 162,   0],
                                 [  0, 242, 255]]
                                )
    def __getitem__(self, index):

        fn_t0 = self.paths['query'][index]
        fn_t1 = self.paths['ref'][index]
        fn_mask = self.paths['GT'][index]
        fn_mask_npy = self.paths['GT'][index].replace('GT','GT_npy')
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
            mask_color = np.asarray(mask_color).astype('uint8').transpose(2, 0, 1)
            mask = self.colormap2classmap(mask_color.transpose(1, 2, 0)) # (h,w,1)
            np.save(fn_mask_npy,mask.numpy())
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        #h, w, c = img_t0.shape
        # w_r = int(256 * max(w / 256, 1))
        # h_r = int(256 * max(h / 256, 1))
        (h_r, w_r) = self.img_size

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')
        mask_r_ = np.asarray(mask_r[:,:,None]).astype('uint8')
        mask_binary = (mask_r_>1).astype('uint8')

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
            mask_binary = self.change_transform(mask_binary)

        return {'source_image': img_t1_r_,
                'target_image': img_t0_r_,
                'source_change': mask_binary.squeeze().int(),
                'target_change': mask_binary.squeeze().int(),
                'flow_map': torch.zeros(2,img_t1_r_.shape[1],img_t1_r_.shape[2]),
                #'source_image_size': (h_r,w_r,3)
                }

    def __len__(self):
        return len(self.paths['GT'])

    def get_random_image(self):
        idx = np.random.randint(0, len(self))
        return self.__getitem__(idx)

    def extract_colors(self, mask_r_):
        mask = mask_r_.transpose(1, 2, 0) # channel last
        colors = mask.reshape(-1, mask.shape[-1])  # (H*W,3) # color channel in rgb order
        unique_colors = self.unique(colors)
        self.colors, idx = np.unique(np.concatenate([unique_colors, self.colors]), return_index=True, axis=0)
        self.colors = self.colors[idx.argsort()]

    def unique(self,array):
            uniq, index = np.unique(array, return_index=True, axis=0)
            return uniq[index.argsort()]

    def colormap2classmap(self, seg_array):
        seg_array_flattened = torch.LongTensor(seg_array.reshape(-1,3))
        seg_map_class_flattened = torch.zeros((seg_array.shape[0],seg_array.shape[1],1)).view(-1,1)
        for cls,color in enumerate(self.colors):
            matching_indices = (seg_array_flattened == torch.LongTensor(color))
            matching_indices = (matching_indices.sum(dim=1)==3)
            seg_map_class_flattened[matching_indices] = cls
        seg_map_class = seg_map_class_flattened.view(seg_array.shape[0],seg_array.shape[1],1)
        return seg_map_class

if __name__ == '__main__':
    dataset = vl_cmu_cd_eval(root='/home/rit/E2EChangeDet/dataset/test_datasets/VL-CMU-CD')
    import matplotlib.pyplot as plt
    i=0
    while(1):
        sample = dataset[i]
        # plt.subplot(411)
        # plt.imshow(mask_color)
        # plt.subplot(412)
        # plt.imshow(mask_binary[0])
        # plt.subplot(413)
        # plt.imshow(img_t0_r_)
        # plt.subplot(414)
        # plt.imshow(img_t1_r_)
        # plt.show()
        # plt.close()
        print(i)
        i+=1
