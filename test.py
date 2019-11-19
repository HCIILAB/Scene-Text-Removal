from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
from datetime import datetime
import time
import logging
from network import set_network
from imagepool import ImagePool
from dataset import load_data,MyDataSet
from vis_dataset import visualize
from mxnet.gluon.data import Dataset, DataLoader
import glob
from mxnet import image,nd
import argparse
def test(args):
	use_gpu = args.gpu
	ctx = mx.gpu(0) if use_gpu else mx.cpu(0)
	img_lists = glob.glob(args.test_image + '/*')
	netG, netD,net, net_label ,trainerG, trainerD, trainerV, trainerL = set_network(args)
	netG.collect_params().reset_ctx(ctx)
	netG.collect_params().load(args.model,ctx = ctx)
	# FPS = 0
	# all_time = 0
	# btic = time.time()
	for i, x in enumerate(img_lists):
		time1 = time.time()
		prefix = x.split('/')[-1].split('.')[0]
		data1 = image.imread(x)
		data = data1.astype(np.float32)/127.5 - 1
		data = image.imresize(data, args.input_size, args.input_size)
		data = nd.transpose(data, (2,0,1))
		data = data.reshape((1,) + data.shape)
		img_name = x.split('/')[-1].split('.')[0]
		real_in = data.as_in_context(ctx)
		# all_time = all_time + time.time()-time1
		# btic = time.time()
		p5,p6,p7,p8,fake_out = netG(real_in)
		# sppeed = time.time() - btic
		# FPS = FPS + sppeed
		# print (FPS,all_time)
		fake_img = fake_out[0]
		predict = ((fake_img.asnumpy().transpose(1, 2, 0) + 1.0).clip(0,2) * 127.5).astype(np.uint8)

		plt.imshow(predict)
		if args.vis:
			plt.show()
		# plt.show()
		prefix = x.split('/')[-1].split('.')[0]
		save_path = args.result + prefix + '.png'
		plt.savefig(save_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--test_image', nargs='?', type=str, default='',    
                        help='Test image path')
    parser.add_argument('--model', nargs='?', type=str, default='',    
                        help='Path to  saved model to restart from')
    parser.add_argument('--input_size', nargs='?', type=int, default=512,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--lr', nargs='?', type=float, default=0.0002, 
                        help='Learning Rate')
    parser.add_argument('--beta', nargs='?', type=float, default=0.0002, 
                        help='beta')
    parser.add_argument('--gpu', nargs='?', type=bool, default=True, 
                        help='use_gpu')
    parser.add_argument('--vis', nargs='?', type=bool, default=True, 
                        help='vis result')
    parser.add_argument('--result', nargs='?', type=str, default='',    
                        help='Path to save resulted images')
    args = parser.parse_args()
    test(args)