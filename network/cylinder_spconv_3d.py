# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn
import torch

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size, val_grid=None, voting_num=4, use_tta=False):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        if use_tta:
            batch_size *= voting_num

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        if use_tta:
            features_ori = torch.split(spatial_features, 1, dim=0)
            fused_predict = features_ori[0][0, :, val_grid[0][:, 0], val_grid[0][:, 1], val_grid[0][:, 2]]
            for idx in range(1, voting_num, 1):
                fused_predict += features_ori[idx][0, :, val_grid[idx][:, 0], val_grid[idx][:, 1], val_grid[idx][:, 2]]
            return fused_predict
        else: 
            return spatial_features
