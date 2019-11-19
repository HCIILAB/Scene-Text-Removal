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
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout,BatchNorm
from mxnet import autograd
import numpy as np
from mxnet.gluon.model_zoo import vision as models
import mxnet
# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)

# Define Unet generator
class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        #Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

# Define the PatchGAN discriminator
class Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1)/2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult))
            if use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print (self.model)
        #print(out)
        return out

class label_Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=1, n_layers=3, use_sigmoid=False, use_bias=False):
        super(label_Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 70
            padding = 24
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=8,
                                  padding=padding, in_channels=in_channels, use_bias=use_bias))
            if use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        return out
class LC(HybridBlock):
    def __init__(self, outer_channels):
        super(LC, self).__init__()
        with self.name_scope():
            channels = int(np.ceil(outer_channels/2))
            self.model = HybridSequential()
            self.model.add(nn.Conv2D(channels,kernel_size=1))
            self.model.add(nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=1))
            self.model.add(nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=1))
    def hybrid_forward(self, F, x):
        out = self.model(x)
        return out
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                      strides=strides)
    def hybrid_forward(self, F, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)
def get_net(style_layers):
    ctx = mx.gpu(0)
    vgg16 = models.vgg19(pretrained=True)
    net = nn.Sequential()
    for i in range(max(style_layers)+1):
        net.add(vgg16.features[i])
    # net.collect_params().reset_ctx(ctx)
    return net
def extract_features(x,in_size = 224):
    B,C,H,W = x.shape
    Img = (x + 1.0)*127.5
    Img_chanells = [nd.expand_dims(Img[:,i,:,:],axis=1) for i in range(3)]
    Img_chanells[0] = (Img_chanells[0]/255 - 0.485)/ 0.229       #subtracted by [103.939, 116.779, 123.68]
    Img_chanells[1] = (Img_chanells[1]/255 - 0.456)/ 0.224         #subtracted by [103.939, 116.779, 123.68]
    Img_chanells[2] = (Img_chanells[2]/255 - 0.406)/ 0.225         #subtracted by [103.939, 116.779, 123.68]
    Img = nd.concat(*Img_chanells,dim=1)
    limx = H - in_size
    limy = W - in_size
    xs = np.random.randint(0,limx,B)
    ys = np.random.randint(0,limy,B)
    lis = [nd.expand_dims(Img[i,:,x:x+in_size,y:y+in_size],axis=0) for i,(x,y) in enumerate(zip(xs,ys))]
    Img_cropped = nd.concat(*lis,dim=0)
    return Img_cropped
class STE(nn.Block):
    """docstring for STE nn.HybridBlock """
    def __init__(self,**kwargs):
        super(STE,self).__init__(**kwargs)
#         self.verbose = verbose
        # self.in_planes = 64
        with self.name_scope():
            self.layer1 = nn.Conv2D(64, kernel_size=4, strides=2,padding=1)
            self.lc1 = LC(64)
            self.conv1 = nn.Conv2D(64,kernel_size=1)
            # self.bn1 = nn.BatchNorm()
            # self.relu_conv1 = nn.Activation(activation='relu')
            self.a1 = nn.MaxPool2D(pool_size=2, strides=2)
            self.a2 = Residual(64)
            self.layer2 = Residual(64)
            self.lc2 = LC(64)
            self.conv2 = nn.Conv2D(64,kernel_size=1)
            # self.bn2 = nn.BatchNorm()
            # self.relu_conv2 = nn.Activation(activation='relu')
            self.b1 = Residual(128, same_shape=False)
            self.layer3 = Residual(128)
            self.lc3 = LC(128)
            self.conv3 = nn.Conv2D(128,kernel_size=1)
            # self.bn3 = nn.BatchNorm()
            # self.relu_conv3 = nn.Activation(activation='relu')
            self.c1 = Residual(256, same_shape=False)
            self.layer4 = Residual(256)
            self.lc4 = LC(256)
            self.conv4 = nn.Conv2D(256,kernel_size=1)
            # self.bn4 = nn.BatchNorm()
            # self.relu_conv4 = nn.Activation(activation='relu')
            self.d1 = Residual(512, same_shape=False)
            self.layer5 = Residual(512)

            # block 6
            # b6 = nn.Sequential()
            # b6.add(
            #     nn.AvgPool2D(pool_size=3),
            #     nn.Dense(num_classes)
            # )
            self.layer6 = nn.Conv2D(2,kernel_size=1)
            self.delayer1 = nn.Conv2DTranspose(256, kernel_size=4, padding=1,strides=2)
            # self.debn1 = nn.BatchNorm()
            self.relu1 = nn.ELU(alpha=1.0)
            # self.relu1 = nn.ELU(alpha=0.2)
            # self.relu1 = nn.ELU(alpha=0.2)
            # self.relu11 = nn.(activation='relu')
            self.relu11 = nn.ELU(alpha=1.0)
            # self.relu11 = nn.ELU(alpha=1.0)
            # mxnet.ndarray.add(lhs, rhs)
            self.delayer2 = nn.Conv2DTranspose(128, kernel_size=4, padding=1,strides=2)
            # self.debn2 = nn.BatchNorm()
            self.relu2 = nn.ELU(alpha=1.0)
            self.relu22 = nn.ELU(alpha=1.0)
            self.delayer3 = nn.Conv2DTranspose(64, kernel_size=4, padding=1,strides=2)
            self.convs_1 = Conv2D(channels=3, kernel_size=1, strides=1, padding=0,use_bias=False)
            # self.debn3 = nn.BatchNorm()
            self.relu3 = nn.ELU(alpha=1.0)
            self.relu33 = nn.ELU(alpha=1.0)
            self.delayer4 = nn.Conv2DTranspose(64, kernel_size=4, padding=1,strides=2)
            self.convs_2 =Conv2D(channels=3, kernel_size=1, strides=1, padding=0,use_bias=False)
            # self.debn4 = nn.BatchNorm()
            self.relu4 = nn.ELU(alpha=1.0)
            self.relu44 = nn.ELU(alpha=1.0)
            self.delayer5 = nn.Conv2DTranspose(3, kernel_size=4, padding=1,strides=2)
            # self.debn5 = nn.BatchNorm()
            self.relu5 = nn.ELU(alpha=1.0)


    def forward(self, x):
        c1 = self.layer1(x)
        lc1 = self.lc1(c1)
        a1 = self.a1(c1)
        a2 = self.a2(a1)
        c2 = self.layer2(a2)
        lc2 = self.lc2(c2)
        b1 = self.b1(c2)
        c3 = self.layer3(b1)
        lc3 = self.lc3(c3)
        C1 = self.c1(c3)
        c4 = self.layer4(C1)
        lc4 = self.lc4(c4)
        d1 = self.d1(c4)
        c5 = self.layer5(d1)
        p51 = self.layer6(c5)
        p5 = self.relu11(self.conv4(lc4) + self.relu1(self.delayer1(p51)))
        p6 = self.relu22(self.conv3(lc3) + self.relu2(self.delayer2(p5)))
        p7 = self.relu33(self.conv2(lc2) + self.relu3(self.delayer3(p6)))
        p7_o = self.convs_1(p7)
        p8 = self.relu44(self.conv1(lc1) + self.relu4(self.delayer4(p7)))
        p8_o = self.convs_2(p8)
        p9 = self.relu5(self.delayer5(p8))
        return p5,p6,p7_o,p8_o,p9
