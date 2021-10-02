import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from glob import glob
import torchvision.transforms as transforms
import albumentations as A

def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','png']])

class vl_cmu_cd(Dataset):

    def __init__(self, root,
                 ):
        super(vl_cmu_cd, self).__init__()
        self.img_t0_root = pjoin(root, 't0')
        self.img_t1_root = pjoin(root, 't1')
        self.img_mask_root = pjoin(root, 'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.png')
        fn_t1 = pjoin(self.img_t1_root, fn + '.png')
        fn_mask = pjoin(self.img_mask_root, fn + '.png')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)
        mask = cv2.imread(fn_mask, 0)

        mask_r = mask[:, :, np.newaxis]

        img_t0_r = np.asarray(img_t0).astype('f').transpose(2, 0, 1)
        img_t1_r = np.asarray(img_t1).astype('f').transpose(2, 0, 1)
        mask_r_ = np.asarray(mask_r > 128).astype('f').transpose(2, 0, 1)


        input_ = torch.from_numpy(np.concatenate((img_t0_r, img_t1_r), axis=0))
        mask_ = torch.from_numpy(mask_r_).long()

        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)


class vl_cmu_cd_eval(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 change_transform=None,
                 ):
        super(vl_cmu_cd_eval, self).__init__()
        self.paths = {'GT':glob(pjoin(root,'*','GT','*.png'))}
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

        w, h, c = img_t0.shape
        w_r = int(256 * max(w / 256, 1))
        h_r = int(256 * max(h / 256, 1))

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
                'source_change': mask_binary.squeeze(),
                'target_change': mask_binary.squeeze(),
                'source_image_size': (h_r,w_r,3)
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
    dataset = vl_cmu_cd_eval(root='/media/rit/GLU-CHANGE-SSD500/dataset/VL-CMU-CD')
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
