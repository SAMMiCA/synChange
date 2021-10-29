import os
import argparse
import glob
from natsort import natsorted
import cv2
import torch
import torchvision
fname_ours = 'vis_ours_pcd'
fname_drta = 'vis_drta_pcd'
fname_cscd = 'vis_cscd_pcd'
parser = argparse.ArgumentParser(description='viz script')
parser.add_argument('--fnames', nargs='+', type=str,
                    default=[fname_ours, fname_drta, fname_cscd])
parser.add_argument('--dnames', nargs='+', type=str,
                    default=['tsunami','gsv'])
args = parser.parse_args()


model_paths = [os.path.join('snapshots',fname) for fname in args.fnames]
for dname in args.dnames:
    t0_images = natsorted(glob.glob(os.path.join(model_paths[0],dname,'t0','*.png')))[::-1]
    t1_images = natsorted(glob.glob(os.path.join(model_paths[0],dname,'t1','*.png')))[::-1]
    pred_on_remapped_images = natsorted(glob.glob(os.path.join(model_paths[0],dname,'pred_on_remapped','*.png')))[::-1]
    gt_on_t1_images = natsorted(glob.glob(os.path.join(model_paths[0],dname,'gt_on_t1','*.png')))[::-1]
    pred1s = natsorted(glob.glob(os.path.join(model_paths[0],dname,'pred_on_t1','*.png')))[::-1]
    pred2s = natsorted(glob.glob(os.path.join(model_paths[1],dname,'pred_on_t1','*.png')))[::-1]
    pred3s = natsorted(glob.glob(os.path.join(model_paths[2],dname,'pred_on_t1','*.png')))[::-1]



    for t0,t1,pred_remap,gt,pred1, pred2, pred3 in zip(t0_images,t1_images,pred_on_remapped_images,gt_on_t1_images,
                                                       pred1s,pred2s,pred3s):
        imgpaths = [t0,t1,pred3,pred2,pred1,pred_remap, gt]
        imgs = []
        for imgpath in imgpaths:
            img = cv2.imread(imgpath)
            img = cv2.resize(img,(256,256))
            imgs.append(torch.FloatTensor(img).permute(2,0,1))
        grid = torchvision.utils.make_grid(imgs,nrow=7,padding=10,pad_value=255)
        grid = grid.permute(1,2,0).numpy()
        savepath = os.path.join('snapshots','merged',dname)
        os.makedirs(savepath,exist_ok=True)
        cv2.imwrite(os.path.join(savepath,gt.split('/')[-1]),grid)

