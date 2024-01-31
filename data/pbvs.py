# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   npbvs.py
@Time    :   2022/3/9 20:54
@Desc    :
"""
import os
import glob
import numpy as np
from data import augment
from torch.utils import data
from utils import imresize, down_sample, torch_psnr

def get_img_list(attr, scale):
    val_list = [str(img_name).zfill(5) + '.npy' for img_name in val1_list]

    root_path = None
    for path in ['/data_c/zhwzhong', '/home/zhwzhong', '/home/temp', '/root/autodl-tmp']:
        tmp_path = '{}/Data'.format(path)
        if os.path.exists(tmp_path):
            root_path = tmp_path
    if attr == 'train':
        img_list = list(glob.glob('{}/PBVS/track1/{}/640_flir_hr/*.npy'.format(root_path, attr)))
        img_list = [img_name for img_name in img_list if os.path.basename(img_name) not in val_list]

    elif attr == 'val':
        img_list = [os.path.join('{}/PBVS/track1/train/640_flir_hr/'.format(root_path), img_name) for img_name in val_list]
        img_list = img_list * 5 if scale == 4 else img_list
    else:
        img_list = glob.glob('{}/PBVS/track1/{}/640_flir_hr/*.npy'.format(root_path, 'test1'))
    return sorted(img_list)


class PBVS(data.Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        self.img_dict = {}
        self.img_list = get_img_list(attr, args.scale)

    def __len__(self):
        return int(self.args.show_every * len(self.img_list)) if self.attr == 'train' else len(self.img_list)

    def __getitem__(self, item):
        item = item % len(self.img_list)
        lr_img, lr_up, gt_img = self.get_img(self.img_list[item])

        if self.attr == 'train':
            scale = self.args.scale
            patch_size = self.args.patch_size // scale
            lr_img, lr_up, gt_img = augment.get_patch(lr_img, lr_up, gt_img, patch_size=patch_size, scale=scale)

            lr_img, lr_up, gt_img = augment.random_rot(lr_img, lr_up, gt_img, hflip=True, rot=True)

        lr_img, lr_up, gt_img = augment.np_to_tensor(lr_img, lr_up, gt_img, input_data_range=255, process_data_range=1)

        img_name = os.path.basename(self.img_list[item])
        return {'img_gt': gt_img, 'lr_up': lr_up, 'img_name': img_name, 'img_lr': lr_img}

    def get_img(self, img_name):
        if img_name not in self.img_dict.keys():
            gt_img = np.load(img_name)
            if self.attr == 'test' and self.args.scale == 4:
                lr_img = np.load(img_name.replace('640_flir_hr', '320_axis_mr'))
            else:
                lr_img = down_sample(gt_img, scale=1 / self.args.scale)

            lr_up = np.expand_dims(imresize(lr_img.astype(float), self.args.scale), 0)

            lr_img = np.expand_dims(lr_img, 0)
            self.img_dict[img_name] = {'lr_up': lr_up, 'gt_img': np.expand_dims(gt_img, 0), 'img_lr': lr_img}
        return self.img_dict[img_name]['img_lr'], self.img_dict[img_name]['lr_up'], self.img_dict[img_name]['gt_img']

