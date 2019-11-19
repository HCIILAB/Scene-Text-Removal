#coding=utf-8
import os
import mxnet as mx
from mxnet import nd
import numpy as np
from matplotlib import pyplot as plt
from mxnet.gluon.data import Dataset, DataLoader
from vis_dataset import visualize
import random
import cv2
pool_size = 50
img_wd = 512
img_ht = 512
def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = nd.image.flip_left_right(imgs[i]).copy()
    return imgs
def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    # print(angle)
    for i in range(len(imgs)):
        img = imgs[i].asnumpy()
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = nd.array(img_rotation)
    return imgs
class MyDataSet(Dataset):
    def __init__(self, root, split, is_transform=False,is_train=True):
        self.root = os.path.join(root, split)
        self.is_transform = is_transform
        self.img_paths = []
        self._img_512 = os.path.join(root, split, 'train_512', '{}.png')
        self._mask_512 = os.path.join(root, split, 'mask_512', '{}.png')
        self._lbl_512 = os.path.join(root, split, 'train_512', '{}.png')
        self._img_256 = os.path.join(root, split, 'train_256', '{}.png')
        self._lbl_256 = os.path.join(root, split, 'train_256', '{}.png')
        self._img_128 = os.path.join(root, split, 'train_128', '{}.png')
        for fn in os.listdir(os.path.join(root, split, 'train_512')):
            if len(fn) > 3 and fn[-4:] == '.png':
                self.img_paths.append(fn[:-4])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path_512 = self._img_512.format(self.img_paths[idx])
        img_path_256 = self._img_256.format(self.img_paths[idx])
        img_path_128 = self._img_128.format(self.img_paths[idx])
        lbl_path_256 = self._lbl_256.format(self.img_paths[idx])
        mask_path_512 = self._mask_512.format(self.img_paths[idx])
        lbl_path_512 = self._lbl_512.format(self.img_paths[idx])
        img_arr_256 = mx.image.imread(img_path_256).astype(np.float32)/127.5 - 1
        img_arr_512 = mx.image.imread(img_path_512).astype(np.float32)/127.5 - 1
        img_arr_128 = mx.image.imread(img_path_128).astype(np.float32)/127.5 - 1
        img_arr_512 = mx.image.imresize(img_arr_512, img_wd * 2, img_ht)
        img_arr_in_512, img_arr_out_512 = [mx.image.fixed_crop(img_arr_512, 0, 0, img_wd, img_ht),
                                        mx.image.fixed_crop(img_arr_512, img_wd, 0, img_wd, img_ht)]
        if os.path.exists(mask_path_512):
            mask_512 = mx.image.imread(mask_path_512)
        else:
            mask_512 = mx.image.imread(mask_path_512.replace(".png",'.jpg',1))
        tep_mask_512 = nd.slice_axis(mask_512, axis=2, begin=0, end=1)/255
        if self.is_transform:
            imgs = [img_arr_out_512, img_arr_in_512, tep_mask_512,img_arr_256,img_arr_128]
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            img_arr_out_512,img_arr_in_512,tep_mask_512,img_arr_256,img_arr_128 = imgs[0], imgs[1], imgs[2], imgs[3],imgs[4]
        img_arr_in_512, img_arr_out_512 = [nd.transpose(img_arr_in_512, (2,0,1)),
                                        nd.transpose(img_arr_out_512, (2,0,1))]
        img_arr_out_256 = nd.transpose(img_arr_256, (2,0,1))
        img_arr_out_128 = nd.transpose(img_arr_128, (2,0,1))
        tep_mask_512 = tep_mask_512.reshape(tep_mask_512.shape[0],tep_mask_512.shape[1],1)
        tep_mask_512 = nd.transpose(tep_mask_512,(2,0,1))
        return img_arr_out_512,img_arr_in_512,tep_mask_512,img_arr_out_256,img_arr_out_128