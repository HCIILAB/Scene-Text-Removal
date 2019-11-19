import os
import mxnet as mx
from mxnet import nd
import numpy as np
from matplotlib import pyplot as plt
from mxnet.gluon.data import Dataset, DataLoader
from vis_dataset import visualize
pool_size = 50
img_wd = 512
img_ht = 512
def load_data(path, batch_size, is_reversed=False):
    img_in_list = []
    img_out_list = []
    mask_list = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith('.png'):
                continue
            img = os.path.join(path, fname)
            # print img
            # raw_input()
            img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
            img_arr = mx.image.imresize(img_arr, img_wd * 2, img_ht)
            # Crop input and output images
            img_arr_in, img_arr_out = [mx.image.fixed_crop(img_arr, 0, 0, img_wd, img_ht),
                                       mx.image.fixed_crop(img_arr, img_wd, 0, img_wd, img_ht)]
            img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2,0,1)),
                                       nd.transpose(img_arr_out, (2,0,1))]
            img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                       img_arr_out.reshape((1,) + img_arr_out.shape)]
            # print (img_arr_in)
            # input(0)
            mask = img.replace('train1','mask',1)
            # print mask
            mask = mx.image.imread(mask)
            # tep_mask = nd.zeros((1,img_ht,img_wd))
            tep_mask = nd.slice_axis(mask, axis=2, begin=0, end=1)/255
            # for k in range(256):
            #     for l in range(256):
            #         if tep_mask[k,l,:] ==0:
            #             print ('yes')
            # print tep_mask
            # plt.imshow(mask.asnumpy())
            # plt.show()
            tep_mask = nd.transpose(tep_mask,(2,0,1))
            tep_mask = tep_mask.reshape((1,) + tep_mask.shape)
            # print tep_mask.shape
            img_in_list.append(img_arr_out if is_reversed else img_arr_in)
            img_out_list.append(img_arr_in if is_reversed else img_arr_out)
            mask_list.append(tep_mask)
    return mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0), nd.concat(*img_out_list, dim=0),nd.concat(*mask_list, dim=0)],
                             batch_size=batch_size)
class MyDataSet(Dataset):
    def __init__(self, root, split, is_train=True):
        self.root = os.path.join(root, split)
        # self.transform = transform
        self.img_paths = []
        self._img_512 = os.path.join(root, split, 'train_512', '{}.png')
        self._mask_512 = os.path.join(root, split, 'mask_512', '{}.png')
        self._lbl_512 = os.path.join(root, split, 'train_512', '{}.png')
        self._img_256 = os.path.join(root, split, 'train_256', '{}.png')
        # self._mask_256 = os.path.join(root, split, 'mask', '{}.png')
        self._lbl_256 = os.path.join(root, split, 'train_256', '{}.png')
        self._img_128 = os.path.join(root, split, 'train_128', '{}.png')
        # self._img_64 = os.path.join(root, split, 'train_64', '{}.png')
        for fn in os.listdir(os.path.join(root, split, 'mask_512')):
            if len(fn) > 3 and fn[-4:] == '.png':
                self.img_paths.append(fn[:-4])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # img_in_list_256 = []
        # img_out_list_256 = []
        # img_out_list_128 = []
        # img_out_list_64 = []
        # mask_list_256 = []
        img_path_512 = self._img_512.format(self.img_paths[idx])
        img_path_256 = self._img_256.format(self.img_paths[idx])
        img_path_128 = self._img_128.format(self.img_paths[idx])
        # img_path_64 = self._img_64.format(self.img_paths[idx])
        # mask_path_256 = self._mask_256.format(self.img_paths[idx])
        lbl_path_256 = self._lbl_256.format(self.img_paths[idx])
        mask_path_512 = self._mask_512.format(self.img_paths[idx])
        lbl_path_512 = self._lbl_512.format(self.img_paths[idx])
        # img_arr_256 = mx.image.imread(img_path_256).astype(np.float32)/255.0
        label_arr_512 = mx.image.imread(lbl_path_512).astype(np.float32)/127.5 - 1
        # img_arr_128 = mx.image.imread(img_path_128).astype(np.float32)/255.0
        img_arr_256 = mx.image.imread(img_path_256).astype(np.float32)/127.5 - 1
        img_arr_512 = mx.image.imread(img_path_512).astype(np.float32)/127.5 - 1
        img_arr_128 = mx.image.imread(img_path_128).astype(np.float32)/127.5 - 1
        # img_arr_64 = mx.image.imread(img_path_64).astype(np.float32)/127.5 - 1
        # img_arr_256 = mx.image.imresize(img_arr_256, img_wd, img_ht/2)
        # img_arr_512 = mx.image.imresize(img_arr_512, img_wd * 2, img_ht)
        # img_arr_128 = mx.image.imresize(img_arr_128, img_wd, img_ht)
        # img_arr_64 = mx.image.imresize(img_arr_64, img_wd, img_ht)
         # Crop input and output images
        img_arr_in_512, img_arr_out_512 = [mx.image.fixed_crop(label_arr_512, 0, 0, img_wd, img_ht),
                                        mx.image.fixed_crop(img_arr_512, img_wd, 0, img_wd, img_ht)]
                                        # img_arr_512]
        img_arr_in_512, img_arr_out_512 = [nd.transpose(img_arr_in_512, (2,0,1)),
                                        nd.transpose(img_arr_out_512, (2,0,1))]
        # img_arr_in_256, img_arr_out_256 = [mx.image.fixed_crop(img_arr_256, 0, 0, img_wd/2, img_ht/2),
                                       # mx.image.fixed_crop(img_arr_256, img_wd/2, 0, img_wd/2, img_ht/2)]
        img_arr_in_256 = nd.transpose(img_arr_256, (2,0,1))
                                       # nd.transpose(img_arr_out_256, (2,0,1))]
        # img_arr_out_128, img_arr_out_64 = [nd.transpose(img_arr_128, (2,0,1)),
        #                                nd.transpose(img_arr_64, (2,0,1))]
        img_arr_out_128 = nd.transpose(img_arr_128, (2,0,1))
        # img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
        #                                img_arr_out.reshape((1,) + img_arr_out.shape)]
        # mask_256 = mx.image.imread(mask_path_256)
        # tep_mask_256 = nd.slice_axis(mask_256, axis=2, begin=0, end=1)/255
        # tep_mask_256 = nd.transpose(tep_mask_256,(2,0,1))
        mask_512 = mx.image.imread(mask_path_512)
        tep_mask_512 = nd.slice_axis(mask_512, axis=2, begin=0, end=1)/255
        tep_mask_512 = nd.transpose(tep_mask_512,(2,0,1))
        # tep_mask = tep_mask.reshape((1,) + tep_mask.shape)
        # visualize(img_arr_out_512)
        # plt.show()
        # visualize(img_arr_in_512)
        # plt.show()
        # visualize(img_arr_out_64)
        # plt.show()
        # raw_input()
        # img = cv2.imread(img_path)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        # mask = np.bitwise_not(mask)
        # lbl = np.bitwise_or(mask, lbl/255)
        # #lbl = lbl/255
        # if not self.transform is None:
        #     img, lbl = self.transform(img, lbl)

        return img_arr_out_512,img_arr_in_512,tep_mask_512,img_arr_in_256,img_arr_out_128
