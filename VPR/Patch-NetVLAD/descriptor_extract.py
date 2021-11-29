import argparse
import configparser
import os
from pathlib import Path

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tools.datasets import PlaceDataset
from models.patch_netvlad.models_generic import get_backend, get_model, get_pca_encoding
from dataloader import ChangeSim

ROOT_DIR = os.getcwd()


def descriptor_extract(eval_set, model, device, opt, config):
    if not os.path.exists(opt.descriptors_dir):
        os.makedirs(opt.descriptors_dir)

    pool_size = int(config['global_params']['num_pcs'])

    # Assuming eval_set is an instance of ChangeSim
    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))

    model.eval()
    with torch.no_grad():
        tqdm.write('======> Extracting Descriptors')
        db_desc = np.empty((len(eval_set), pool_size), dtype=np.float32)
        start_ind = 0
        end_ind = 0
        for iter, (rgbs, clable, paths) in enumerate(tqdm(test_data_loader)):
            # list of rgb file names
            batch_size = len(paths)
            end_ind += batch_size
            rgb_indicies = np.arange(start_ind, end_ind)
            start_ind += batch_size
            
            enc = model.encoder(rgbs.to(device))
                
            if config['global_params']['pooling'].lower() == 'patchnetvlad':
                # vlad_local.shape = P * [( N, K*d, P_n)], P = # of heterogeneous patches, P_n = # of patches
                # vlad_global.shape = (N, K*d)
                vlad_local, vlad_global = model.pool(enc)
            
                vlad_global_pca = get_pca_encoding(model, vlad_global)
                db_desc[rgb_indicies, :] = vlad_global_pca.detach().cpu().numpy()
                
                for patch_size_idx, this_local in enumerate(vlad_local):
                    this_patch_size = model.pool.patch_sizes[patch_size_idx]
                    
                    # this_local_pca.shape = (N*P_n, num_pcs)
                    this_local_pca = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1)))
                    # this_local_pca.shape = (N, num_pcs, P_n)
                    this_local_pca = this_local_pca.reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
                    db_desc_patches = this_local_pca.detach().cpu().numpy().copy()
                    
                    for i in range(batch_size):
                        if opt.dataset == 'vlcmucd':
                            seq_index = Path(paths[i]).parents[1].parts[-1]
                            frame_index = Path(paths[i]).stem[2:]
                            img_name = seq_index + '_' + frame_index
                        else:
                            img_name = Path(paths[i]).stem

                        filename = opt.descriptors_dir + '/' + 'patch_desc' + \
                            '_' + 'psize{}_'.format(this_patch_size) + \
                            img_name + '.npy'
                        np.save(filename, db_desc_patches[i])
            else:
                vlad_global = model.pool(enc)
                vlad_global_pca = get_pca_encoding(model, vlad_global)
                db_desc[rgb_indicies, :] = vlad_global_pca.detach().cpu().numpy()
    
    global_desc_filename = os.path.join(opt.descriptors_dir, 'global_desc.npy')
    np.save(global_desc_filename, db_desc)
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VPR-Models-Descriptor-Extract')
    parser.add_argument('--config_path', type=str, 
                        default=os.path.join(ROOT_DIR, 'configs/performance.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--dataset_root_dir', type=str, default='',
                        help='If the files in dataset_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--query_data_dir', type=str,
                        help='Directory of query data', required=True)
    parser.add_argument('--ref_data_dir', type=str,
                        help='Directory of ref data', required=True)
    parser.add_argument('--descriptors_dir', type=str, default=os.path.join(ROOT_DIR, 'output_descriptors'),
                        help='Path to store all patch-netvlad descriptors')
    parser.add_argument('--out_ref', type=bool, default=False)
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    dataset_cs = ChangeSim(q_path=opt.query_data_dir,
                           r_path=opt.ref_data_dir,
                           crop_size=(int(config['extract']['imageresizew']), int(config['extract']['imageresizeh'])), 
                           num_classes=5, out_ref=opt.out_ref, set='test', dataset=opt.dataset)

    # must resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    if os.path.isfile(resume_ckpt):
        print(f"=> loading checkpoint '{resume_ckpt}'")
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, opt, config['global_params'], append_pca_layer=True)
        model.load_state_dict(checkpoint['state_dict'])
        
        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

       
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    descriptor_extract(dataset_cs, model, device, opt, config)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('\n\nDone. Finished extracting and saving descriptors')