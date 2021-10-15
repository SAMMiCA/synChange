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

class pcd_5fold(Dataset):
    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 co_transform = None,
                 change_transform=None,
                 fold=0,
                 split='train', # train or test,
                 img_size = (520,520)
    ):
        super(pcd_5fold, self).__init__()
        fold_name = 'set{}'.format(str(fold))
        root = pjoin(root,fold_name,split)
        self.paths = {'GT':glob(pjoin(root,'mask','*.png'))}

        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('mask','t1')
            query_path = query_path.replace('.png','.jpg')
            query_paths.append(query_path)
            ref_path = gtpath.replace('mask','t0')
            ref_path = ref_path.replace('.png','.jpg')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.co_transform = co_transform
        self.change_transform = change_transform
        self.img_size = img_size


    def __getitem__(self, index):

        fn_t0 = self.paths['ref'][index]
        fn_t1 = self.paths['query'][index]
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
            mask = np.asarray(mask_color)[:,:,0]==255
            mask = mask.astype(np.uint8)
            np.save(fn_mask_npy,mask)
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        (h_r, w_r) = self.img_size

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')

        if self.co_transform is not None:
            co_trmd = self.co_transform(image=np.dstack([img_t0_r_,img_t1_r_]),mask=mask_r)
            img_t0_r_,img_t1_r_ = co_trmd['image'][:,:,:3], co_trmd['image'][:,:,3:]
            mask_r = co_trmd['mask']

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
            mask_r = self.change_transform(mask_r[:,:,None])

        return {'source_image': img_t0_r_,
                'target_image': img_t1_r_,
                'source_change': mask_r.squeeze().int(),
                'target_change': mask_r.squeeze().int(),
                #'source_image_size': (h_r,w_r,3),
                'flow_map': torch.zeros(2, img_t1_r_.shape[1], img_t1_r_.shape[2]),
                'correspondence_mask': torch.ones_like(mask_r.squeeze()).numpy().astype(np.bool),
                'use_flow': torch.zeros(1),
                'disable_flow': torch.ones(1),

                }

    def __len__(self):
        return len(self.paths['GT'])



class gsv_eval(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 co_transform = None,
                 change_transform=None,
                 split='train',
                 img_size = (256,512)
                 ):
        super(gsv_eval, self).__init__()
        self.paths = {'GT':glob(pjoin(root,'ground_truth','*.bmp'))}
        self.split = split
        self.paths['GT'] = self.paths['GT'][:80] if self.split == 'train' else self.paths['GT'][80:]
        self.img_size = img_size
        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('ground_truth','t1')
            query_path = query_path.replace('.bmp','.jpg')
            query_paths.append(query_path)
            ref_path = gtpath.replace('ground_truth','t0')
            ref_path = ref_path.replace('.bmp','.jpg')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.co_transform = co_transform
        self.change_transform = change_transform
    def __getitem__(self, index):

        fn_t0 = self.paths['ref'][index]
        fn_t1 = self.paths['query'][index]
        fn_mask = self.paths['GT'][index]
        fn_mask_npy = self.paths['GT'][index].replace('ground_truth','GT_npy')
        fn_mask_npy = fn_mask_npy.replace('bmp','npy')

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
            mask = np.asarray(mask_color)[:,:,0]!=255
            mask = mask.astype(np.uint8)
            np.save(fn_mask_npy,mask)
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        w, h, c = img_t0.shape
        # w_r = int(256 * max(w / 256, 1))
        # h_r = int(256 * max(h / 256, 1))
        (w_r, h_r) = self.img_size

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')

        if self.co_transform is not None:
            co_trmd = self.co_transform(image=np.dstack([img_t0_r_,img_t1_r_]),mask=mask_r)
            img_t0_r_,img_t1_r_ = co_trmd['image'][:,:,:3], co_trmd['image'][:,:,3:]
            mask_r = co_trmd['mask']

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
            mask_r = self.change_transform(mask_r[:,:,None])

        img_t1_r_ = torch.cat([img_t1_r_,img_t1_r_],dim=1)
        img_t0_r = torch.cat([img_t0_r_,img_t0_r_],dim=1)
        mask_r = torch.cat([mask_r.squeeze(),mask_r.squeeze()],dim=0)
        return {'source_image': img_t0_r,
                'target_image': img_t1_r_,
                'source_change': mask_r.int(),
                'target_change': mask_r.int(),
                # 'source_image_size': (h_r,w_r,3),
                'flow_map': torch.zeros(2, img_t1_r_.shape[1], img_t1_r_.shape[2]),
                'correspondence_mask': torch.ones_like(mask_r.squeeze()).numpy().astype(np.bool),
                'use_flow': torch.zeros(1),
                'disable_flow': torch.ones(1),

                }

        # return {'source_image': img_t1_r_,
        #         'target_image': img_t0_r_,
        #         'source_change': mask_r.squeeze(),
        #         'target_change': mask_r.squeeze(),
        #         'source_image_size': (h_r,w_r,3)
        #         }

    def __len__(self):
        return len(self.paths['GT'])


class tsunami_eval(Dataset):

    def __init__(self, root,
                 source_image_transform=None, target_image_transform=None,
                 co_transform = None,
                 change_transform=None,
                 split='train',
                 img_size=(256, 512)

                 ):
        super(tsunami_eval, self).__init__()
        self.paths = {'GT':glob(pjoin(root,'mask','*.png'))}
        self.split = split
        self.img_size = img_size
        self.paths['GT'] = self.paths['GT'][:80] if self.split == 'train' else self.paths['GT'][80:]

        query_paths, ref_paths = [],[]
        for gtpath in self.paths['GT']:
            query_path = gtpath.replace('mask','t1')
            query_path = query_path.replace('.png','.jpg')
            query_paths.append(query_path)
            ref_path = gtpath.replace('mask','t0')
            ref_path = ref_path.replace('.png','.jpg')
            ref_paths.append(ref_path)
        self.paths['query'] = query_paths
        self.paths['ref'] = ref_paths
        # self.colors = np.array([[0,0,0]],dtype=np.uint8)
        self.co_transform = co_transform
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.change_transform = change_transform

    def __getitem__(self, index):

        fn_t0 = self.paths['ref'][index]
        fn_t1 = self.paths['query'][index]
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
            mask = np.asarray(mask_color)[:,:,0]==255
            mask = mask.astype(np.uint8)
            np.save(fn_mask_npy,mask)
            mask = np.load(fn_mask_npy) # (h,w,1)

        else:
            mask = np.load(fn_mask_npy) # (h,w,1)
            mask_color = np.zeros([3,mask.shape[0],mask.shape[1]])

        img_t0 = cv2.imread(fn_t0, 1)
        img_t0 = cv2.cvtColor(img_t0,cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(fn_t1, 1)
        img_t1 = cv2.cvtColor(img_t1,cv2.COLOR_BGR2RGB)

        w, h, c = img_t0.shape
        # w_r = int(256 * max(w / 256, 1))
        # h_r = int(256 * max(h / 256, 1))
        (w_r, h_r) = self.img_size

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r),interpolation=cv2.INTER_NEAREST)
        img_t0_r_ = np.asarray(img_t0_r).astype('uint8')
        img_t1_r_ = np.asarray(img_t1_r).astype('uint8')

        if self.co_transform is not None:
            co_trmd = self.co_transform(image=np.dstack([img_t0_r_,img_t1_r_]),mask=mask_r)
            img_t0_r_,img_t1_r_ = co_trmd['image'][:,:,:3], co_trmd['image'][:,:,3:]
            mask_r = co_trmd['mask']

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
            mask_r = self.change_transform(mask_r[:,:,None])

        img_t1_r_ = torch.cat([img_t1_r_,img_t1_r_],dim=1)
        img_t0_r = torch.cat([img_t0_r_,img_t0_r_],dim=1)
        mask_r = torch.cat([mask_r.squeeze(),mask_r.squeeze()],dim=0)
        return {'source_image': img_t0_r,
                'target_image': img_t1_r_,
                'source_change': mask_r.int(),
                'target_change': mask_r.int(),
                # 'source_image_size': (h_r,w_r,3),
                'flow_map': torch.zeros(2, img_t1_r_.shape[1], img_t1_r_.shape[2]),
                'correspondence_mask': torch.ones_like(mask_r.squeeze()).numpy().astype(np.bool),
                'use_flow': torch.zeros(1),
                'disable_flow': torch.ones(1),

                }

    def __len__(self):
        return len(self.paths['GT'])

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float=get_float

    def __call__(self, array):

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if self.get_float:
            return tensor.float()
        else:
            return tensor

if __name__ == '__main__':
    # co_transform = A.Compose([
    #     A.RandomCrop(height=224,width=224)
    # ])
    for split in ('train','test'):
        for fold in (0,1,2,3,4):
            dataset = pcd_5fold(root='/home/rit/E2EChangeDet/dataset/test_datasets/pcd_5cv',
                                change_transform=transforms.Compose([ArrayToTensor()]),
                                split=split,
                                fold=fold)
            import matplotlib.pyplot as plt
            for i,sample in enumerate(dataset):
                #sample = dataset[i]
                # import pdb; pdb.set_trace()
                # plt.subplot(311)
                # plt.imshow(sample['target_change'])
                # plt.subplot(312)
                # plt.imshow(sample['source_image'])
                # plt.subplot(313)
                # plt.imshow(sample['target_image'])
                # #plt.show()
                # plt.close()
                print(i)
                i+=1
