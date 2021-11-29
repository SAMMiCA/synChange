import json
import os
from pathlib import Path
import glob
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a json file \
                                     containing the information of the used dataset')
    parser.add_argument('--query_dir', type=str,
                        required=True,
                        help='The directory of the query dataset')
    parser.add_argument('--ref_dir', type=str,
                        required=True,
                        help='The directory of the reference dataset')
    parser.add_argument('--gt_path', type=str,
                        default=None,
                        help='The path of the GT file, default=None (for this case, coordinate of \
                              each image is its index')
    parser.add_argument('--positive_threshold', type=int,
                        default=2,
                        help='The threshold discrepancy of the displacement (or frame)')
    parser.add_argument('--dataset', type=str,
                        required=True,
                        help='Name of the dataset')
    
    opt = parser.parse_args()
    print(opt)
    ds_name = opt.dataset
    
    if ds_name == 'vlcmucd':
        root_dir = opt.query_dir
        db_list = glob.glob(os.path.join(root_dir, "*/RGB/1_*"))
        q_list = glob.glob(os.path.join(root_dir, "*/RGB/2_*"))
        num_q, num_db = len(q_list), len(db_list)
        q_coord = [(5 * int(Path(q_p).parents[1].parts[-1]), int(Path(q_p).stem[2:])) for q_p in q_list]
        db_coord = [(5 * int(Path(db_p).parents[1].parts[-1]), int(Path(db_p).stem[2:])) for db_p in db_list]
    elif ds_name in ['changesim', 'tsunami', 'gsv']:
        q_list = glob.glob(os.path.join(opt.query_dir, '*'))
        db_list = glob.glob(os.path.join(opt.ref_dir, '*'))
        num_q, num_db = len(q_list), len(db_list)
        q_coord = [(int(Path(q_path).stem), 0) for q_path in q_list]
        db_coord = [(int(Path(db_path).stem), 0) for db_path in db_list]
    
    json_dict = {'q_list': q_list, 'db_list': db_list, 
                 'q_coord': q_coord, 'db_coord': db_coord,
                 'num_q': num_q, 'num_db': num_db,
                 'thr': opt.positive_threshold}
    
    with open(opt.dataset + '.json', 'w') as f:
        json.dump(json_dict, f)
    
    