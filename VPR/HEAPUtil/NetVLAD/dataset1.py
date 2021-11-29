import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py
import json

def input_transform():
    return transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_val_set(root_dir, dataset):

    if not exists(root_dir):
        raise FileNotFoundError('root_dir:{} does not exist'.format(root_dir))

    structFile = join(root_dir, dataset+'.json')

    return WholeDatasetFromStruct(structFile, root_dir,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
    'posDistThr'])

def parse_dbStruct(path):
    with open(path, 'r') as f:
        ds = json.load(f)

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = ds['db_list']
    locDb = ds['db_coord']

    qImage = ds['q_list']
    locQ = ds['q_coord']

    numDb = ds['num_db']
    numQ = ds['num_q']

    posDistThr = ds['thr']

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, root_dir, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [] + self.dbStruct.dbImage
            
        if not onlyDB:
            self.images += self.dbStruct.qImage

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.locDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives