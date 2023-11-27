# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py
import errno
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, test_dataset_loader, test_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=True)

    output_path = '/content/drive/MyDrive/Colab_Notebooks/SJSU/IAS/Final_Project/PVKD/out_cyl/test'
    voting_num = 4

    if True:
        print('*'*80)
        print('Generate predictions for test split')
        print('*'*80)
        pbar = tqdm(total=len(test_dataset_loader))
        time.sleep(10)
        if True:
            if True:
                my_model.eval()
                with torch.no_grad():
                    for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(
                            test_dataset_loader):
                        test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          test_pt_fea]
                        test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]

                        predict_labels = my_model(test_pt_fea_ten, test_grid_ten, val_batch_size, test_grid, voting_num, use_tta=True)
                        predict_labels = torch.argmax(predict_labels, dim=0).type(torch.uint8)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        test_pred_label = np.expand_dims(predict_labels,axis=1)
                        save_dir = test_pt_dataset.im_idx[test_index[0]]
                        _,dir2 = save_dir.split('/sequences/',1)
                        new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
                        if not os.path.exists(os.path.dirname(new_save_dir)):
                            try:
                                os.makedirs(os.path.dirname(new_save_dir))
                            except OSError as exc:
                                if exc.errno != errno.EEXIST:
                                    raise
                        test_pred_label = test_pred_label.astype(np.uint32)
                        test_pred_label.tofile(new_save_dir)
                        pbar.update(1)
                del test_grid, test_pt_fea, test_grid_ten, test_index
        pbar.close()
        print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
        print('Remapping script can be found in semantic-kitti-api.')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='/content/drive/MyDrive/Colab_Notebooks/SJSU/IAS/Final_Project/PVKD/config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
