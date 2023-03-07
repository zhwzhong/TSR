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
from data.extra import a
from torch.utils import data
from utils import imresize, down_sample, torch_psnr


val1_list = [11, 24, 25, 644, 959, 220, 316, 166, 139, 349, 400, 404, 638, 680, 683, 715, 752, 785, 1, 613]
val2_list = [109, 120, 142, 150, 189, 295, 710, 794, 912, 925, 674, 704, 712, 711, 743, 751, 800, 857, 968, 998]


def get_img_list(attr, scale, extra_data):
    val_list = [str(img_name).zfill(5) + '.npy' for img_name in val1_list]

    root_path = None
    for path in ['/data_c/zhwzhong', '/home/zhwzhong', '/home/temp', '/root/autodl-tmp']:
        tmp_path = '{}/Data'.format(path)
        if os.path.exists(tmp_path):
            root_path = tmp_path
    if attr == 'train':
        img_list = list(glob.glob('{}/PBVS/track1/{}/640_flir_hr/*.npy'.format(root_path, attr)))
        img_list = [img_name for img_name in img_list if os.path.basename(img_name) not in val_list]
        if extra_data:
            # 只有新加的数据
            # img_list = ['{}/PBVS/track1/new/640_flir_hr/{}'.format(root_path, name) for name in a]
            # img_list.extend(['{}/PBVS/track1/new/640_flir_hr/{}'.format(root_path, name) for name in a])
            img_list.extend(list(glob.glob('{}/PBVS/track1/new/640_flir_hr/*.npy'.format(root_path))))

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
        self.img_list = get_img_list(attr, args.scale, args.extra_data)

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


# from options import args
# args.scale = 4
# args.show_every = 1
# print(args)
# from torch.utils.data import DataLoader
# data = DataLoader(PBVS(args, 'test'), batch_size=1)
# print('Number of data {}'.format(len(data)))
# sum_psnr = []
# for _, sample in enumerate(data):
#     _, _, h, w = sample['img_gt'].shape
#     # if h != 256 or w != 256:
#     #     print(sample['lr_up'].shape, sample['img_gt'].shape, sample['img_name'][0])
#     psnr = torch_psnr(sample['lr_up'], sample['img_gt'], data_range=1, border=0)
#     sum_psnr.append(psnr)
#     # print( psnr)
# print(np.mean(sum_psnr))