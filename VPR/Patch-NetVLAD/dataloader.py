import os
import os.path as osp
import copy
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from PIL import Image
from utils import Object_Labeling
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.transforms as transforms
import glob
import pdb
import matplotlib.pyplot as plt
import tkinter
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('TkAgg')


def input_transform(resize=(480, 640), set='train'):
    if set == 'train':
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.25), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class ChangeSim(data.Dataset):
    def __init__(self, q_path, r_path, mode='vpr',
                 crop_size=(320, 240), num_classes=5, set='train', out_ref=False, dataset='changesim'):
        """
        ChangeSim Dataloader
        Please download ChangeSim Dataset in https://github.com/SAMMiCA/ChangeSim

        Args:
            crop_size (tuple): Image resize shape (H,W) (default: (320, 240))
            num_classes (int): Number of target change detection class
                               5 for multi-class change detection
                               2 for binary change detection (default: 5)
            set (str): 'train' or 'test' (defalut: 'train')
        """
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.set = set
        self.mode = mode
        self.out_ref = out_ref
        self.blacklist=[]
        train_list = ['Warehouse_0', 'Warehouse_1', 'Warehouse_2', 'Warehouse_3', 'Warehouse_4', 'Warehouse_5']
        # test_list = ['Warehouse_6', 'Warehouse_7', 'Warehouse_8', 'Warehouse_9']
        test_list = ['Warehouse_6']
        self.query_images = []
        self.ref_images = []
        if set == 'train':
            for map in train_list:
                self.query_images += glob.glob('../../../Query/Query_Seq_Train/' + map + '/Seq_0/rgb/*.png')
                # self.query_images += glob.glob('../../../Query/Query_Seq_Train/' + map + '/Seq_1/rgb/*.png')
        elif set == 'test':
            if dataset in ['changesim', 'tsunami', 'gsv']:
                self.query_images += glob.glob(os.path.join(q_path, '*'))
                self.ref_images += glob.glob(os.path.join(r_path, '*'))
                self.q_coord = [(int(Path(q_p).stem), 0) for q_p in self.query_images]
                self.db_coord = [(int(Path(db_p).stem), 0) for db_p in self.ref_images]
                self.thr = 2
            elif dataset == 'vlcmucd':
                root_dir = q_path
                self.ref_images = glob.glob(os.path.join(root_dir, "*/RGB/1_*"))
                self.query_images = glob.glob(os.path.join(root_dir, "*/RGB/2_*"))
                self.q_coord = [(5 * int(Path(q_p).parents[1].parts[-1]), 0) 
                           for q_p in self.query_images]
                self.db_coord = [(5 * int(Path(db_p).parents[1].parts[-1]), 0) 
                            for db_p in self.ref_images]
                self.thr = 2
            
        # if not max_iters == None:
        #     self.query_images = self.query_images * int(np.ceil(float(max_iters) / len(self.query_images)))
        #     self.query_images = self.query_images[:max_iters]

        if mode == 'vpr':
            pass
        else:
            self.seg = Object_Labeling.SegHelper(idx2color_path='./utils/idx2color.txt', num_class=self.num_classes)
        
        #### Transform((H, W), set='train' or 'test') ####
        self.my_transform = input_transform((self.crop_size[1], self.crop_size[0]), self.set)
        self.images = self.ref_images + self.query_images
        self.numDb = len(self.ref_images)
        self.numQ = len(self.query_images)
        
        self.positives = None
        self.distances = None

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        # Train set
        if self.set == 'train':
            loss = nn.L1Loss()
            while True:
                if index in self.blacklist:
                    index=random.randint(0,self.__len__()-1)
                    continue

                test_rgb_path = self.query_images[index]
                file_idx = test_rgb_path.split('/')[-1].split('.')[0]  # ~~ of ~~.png

                ref_pose_find_path = test_rgb_path.replace(f'rgb/{file_idx}.png',f't0/idx/{file_idx}.txt')
                f = open(ref_pose_find_path,'r',encoding='utf8')
                ref_pose_idx = int(f.readlines()[0])
                g2o_path = test_rgb_path.replace('/Query/Query_Seq_Train','/Reference/Ref_Seq_Train').replace(f'rgb/{file_idx}.png',f'raw/poses.g2o')
                with open(g2o_path,'r',encoding = 'utf8') as f2:
                    while True:
                        line = f2.readline()
                        try:
                            if line.split()[0] == 'VERTEX_SE3:QUAT' and int(line.split()[1]) == ref_pose_idx:
                                ref_pose = line.split()[2:]
                        except:
                            break
                ref_pose = torch.from_numpy(np.array(ref_pose).astype(float))
                change_pose_path = test_rgb_path.replace(f'rgb/{file_idx}.png',f'pose/{file_idx}.txt')
                with open(change_pose_path,'r',encoding='utf8') as f3:
                    change_pose = f3.readline().split()
                    change_pose = torch.from_numpy(np.array(change_pose).astype(float))

                distance = loss(ref_pose.cuda(),change_pose.cuda())
                if distance.item()<0.5:
                    break
                else:
                    self.blacklist.append(index)
                    index=random.randint(0,self.__len__()-1)
        # Test set
        else:
            test_rgb_path = self.query_images[index]

        # Get File Paths
        if self.mode == 'vpr':
            ref_rgb_path = self.ref_images[index]
        else:
            test_depth_path = test_rgb_path.replace('rgb', 'depth')
            ref_rgb_path = test_rgb_path.replace('rgb', 't0/rgb') 
            ref_depth_path = test_rgb_path.replace('rgb', 't0/depth')
            change_segmentation_path = test_rgb_path.replace('rgb', 'change_segmentation')

        test_rgb = Image.open(test_rgb_path)
        ref_rgb = Image.open(ref_rgb_path)

        # Resize, RGB, Color Transform, and Normalization for train set
        # For test set, skip Color transform
        test_rgb = self.my_transform(test_rgb)
        ref_rgb = self.my_transform(ref_rgb)

        # Change Label
        if self.mode == 'vpr':
            labels = np.zeros(len(self.query_images))
            label = labels[index]
        else:
            change_label = Image.open(change_segmentation_path)
            change_label = change_label.resize(self.crop_size, Image.NEAREST)
            change_label_mapping = np.asarray(change_label).copy()
            change_mapping = self.seg.colormap2classmap(change_label_mapping)
            label = change_mapping.permute(2,0,1).squeeze(0).long().cpu().long()

            #### Binarization ####
            if self.num_classes == 2:
                label[label > 0] = 1

            # if (label > 5).sum() > 0:
            #     print(image_path)

        # Horizontal Flip
        if self.set == 'train' and np.random.rand() <= 0.5:
            test_rgb = np.asarray(test_rgb)
            test_rgb = test_rgb[:, :, ::-1]
            test_rgb = np.ascontiguousarray(test_rgb)
            test_rgb = torch.from_numpy(test_rgb)

            ref_rgb = np.asarray(ref_rgb)
            ref_rgb = ref_rgb[:, :, ::-1]
            ref_rgb = np.ascontiguousarray(ref_rgb)
            ref_rgb = torch.from_numpy(ref_rgb)

            label = np.asarray(label)
            label = label[:, ::-1]
            label = np.ascontiguousarray(label)
            label = torch.from_numpy(label)
            
        rgb = ref_rgb if self.out_ref else test_rgb
    
        return rgb, label, test_rgb_path
    
    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_coord)

            self.distances, self.positives = knn.radius_neighbors(self.q_coord, radius=self.thr)

        return self.positives


if __name__ == '__main__':
    dst = ChangeSim(crop_size=(320, 240), num_classes=5, set='train')
    dataloader = data.DataLoader(dst, batch_size=4, num_workers=2)
    for i, data in enumerate(dataloader):
        imgs, labels, path = data
