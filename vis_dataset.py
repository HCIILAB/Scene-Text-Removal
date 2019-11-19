import mxnet as mx
# from dataset import load_data
from matplotlib import pyplot as plt
import numpy as np
epochs = 100
batch_size = 10

use_gpu = True
ctx = mx.gpu(0) if use_gpu else mx.cpu()

lr = 0.0002
beta1 = 0.5
lambda1 = 100

pool_size = 50
img_wd = 512
img_ht = 512
dataset = '/media/zst/d619af75-82b1-47c0-b0b8-2083613e7fb8/home/zst/Documents/ID-CGAN/data'
train_img_path = '%s/train' % (dataset)
val_img_path = '%s/val' % (dataset)
# dataset = 'facades'
# train_data = load_data(train_img_path, batch_size, is_reversed=True)
# val_data = load_data(val_img_path, batch_size, is_reversed=True)
def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    # plt.show()
    # plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 0.0) * 255.0).astype(np.uint8))
    plt.axis('off')

def preview_train_data():
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2,4,i+1)
        visualize(img_in_list[i])
        plt.subplot(2,4,i+5)
        visualize(img_out_list[i])
    plt.show()

# preview_train_data()
