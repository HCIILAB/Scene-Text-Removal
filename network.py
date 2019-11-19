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
from mxnet import init
from model import UnetGenerator,Discriminator,STE,label_Discriminator
from mxnet.gluon.model_zoo import vision as models
ctx = mx.gpu(0)
def get_net(pretrained_net,style_layers):
    net = nn.Sequential()
    for i in range(max(style_layers)+1):
        net.add(pretrained_net.features[i])
    return net
def param_init(param):
    # ctx = mx.cpu()
    if param.name.find('ste0_conv0') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('ste0_instancenorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
    # elif param.name.find('batchnorm1') != -1:
    #     param.initialize(init=mx.init.Zero(), ctx=ctx)
        # if param.name.find('gamma') != -1:
        #     param.set_data(nd.random_normal(1, 0.02, param.data().shape))
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))

def network_init(net):
    for param in net.collect_params().values():
        param_init(param)

def set_network(args):
    style_layers = [4,9,16]
    # Pixel2pixel networks
    # netG = UnetGenerator(in_channels=3, num_downs=8)
    net_label = label_Discriminator(in_channels=1,use_sigmoid=False)
    netG = STE()
    netD = Discriminator(in_channels=6,use_sigmoid=False)
    netvgg = models.vgg16(pretrained=True)
    net = get_net(netvgg,style_layers)
    net.collect_params().reset_ctx(ctx)
    # net.collect_params().setattr('grad_req', 'null')
    # Initialize parameters
    netG.initialize(ctx=ctx,init=init.Xavier())
    if args.model:
        netG.collect_params().load(args.model,ctx = ctx)
    netG.collect_params().reset_ctx(ctx)
    network_init(netD)
    net_label.initialize(ctx=ctx,init=mx.initializer.One())
    net_label.collect_params().setattr('grad_req', 'null')

    net_label.collect_params().reset_ctx(ctx)
    # net.collect_params().setattr('grad_req', 'null')
    # net.collect_params().reset_ctx(ctx)
    # trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': args.lr, 'beta': args.beta})
    # trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': args.lr, 'beta': args.beta})
    # trainerV = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': args.lr, 'beta': args.beta})
    # trainerL = gluon.Trainer(net_label.collect_params(), 'adam', {'learning_rate': args.lr, 'beta': args.beta})
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': args.lr})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': args.lr})
    trainerV = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': args.lr})
    trainerL = gluon.Trainer(net_label.collect_params(), 'adam', {'learning_rate': args.lr})
    return netG, netD,net,net_label,trainerG, trainerD,trainerV, trainerL

# Loss
# GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
# L1_loss = gluon.loss.L1Loss()
#
# netG, netD, net, trainerG, trainerD trainerV = set_network()
