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
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
from datetime import datetime
import time
import logging
from network import set_network
from imagepool import ImagePool
#from dataset import load_data
from vis_dataset import visualize
from mxnet.gluon.data import Dataset, DataLoader
from text2 import MyDataSet
import argparse
def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()
def extract_features(x, style_layers,net,ctx,in_size = 224):
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
    styles = []
    for k in range(len(net)):
        Img_cropped = net[k](Img_cropped.as_in_context(ctx))
        if k in style_layers:
            styles.append(Img_cropped)
    return styles
def gram(x):
    c = x.shape[1]
    n = x.size / x.shape[1]
    y = x.reshape((c, int(n)))
    return nd.dot(y, y.T) / n
def style_loss(yhat, y):
    return nd.abs(gram(yhat) - gram(y)).mean()
def cal_loss_style(hout,hcomp,hgt):
    for i in range(3):
        if i==0:
            L_style_out = style_loss(hout[0],hgt[0])
            L_style_comp = style_loss(hcomp[0],hgt[0])
        else:
            L_style_out = L_style_out + style_loss(hout[i],hgt[i])
            L_style_comp = L_style_comp + style_loss(hcomp[i],hgt[i])
    return L_style_comp + L_style_out
def calc_loss_perceptual(hout,hcomp,hgt):
    for j in range(3):
        if j == 0:
            loss = nd.abs(hout[0]-hgt[0]).mean()
            loss = loss + nd.abs(hcomp[0]-hgt[0]).mean()
        else:
            loss = loss + nd.abs(hout[j]-hgt[j]).mean()
            loss = loss + nd.abs(hcomp[j]-hgt[j]).mean()
    return loss
def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())
def train(args):
    use_gpu = args.gpu
    ctx = mx.gpu(0) if use_gpu else mx.cpu()
    pool_size = 50
    lambda1 = 100
    img_wd = args.img_size
    img_ht = args.img_size
    style_layers = [4,9,16]
    my_train = MyDataSet(args.trainset_path, '')
    train_loader = DataLoader(my_train, batch_size=args.batch_size, shuffle=True, last_batch='rollover')
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L1Loss()
    netG, netD, net, net_label,trainerG, trainerD,trainerV, trainerL = set_network(args)
    image_pool = ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(filename = 'pixel2pixel.log',level=logging.DEBUG)
    for epoch in range(args.n_epoch):
        tic = time.time()
        btic = time.time()
        iter = 0
        # print(trainerG.learning_rate)
        if epoch > 0 and epoch % 200 == 0:
            trainerG.set_learning_rate(trainerG.learning_rate * 0.2)
            trainerD.set_learning_rate(trainerD.learning_rate * 0.2)
            trainerV.set_learning_rate(trainerD.learning_rate * 0.2)
        # print(trainerG.learning_rate)
        for data, label, mask, data_256, data_128 in train_loader:
            batch_size =data.shape[0]
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = data.as_in_context(ctx)
            real_out = label.as_in_context(ctx)
            real_out_256 = data_256.as_in_context(ctx)
            real_out_128 = data_128.as_in_context(ctx)
            mask = mask.as_in_context(ctx).astype('float32')
            mask_b = mask.asnumpy().astype(bool)
            mask = mask.astype('float32')
            _,_,_,_, fake_out = netG(real_in)
            fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Use image pooling to utilize history images
                mask_patch = 1 - net_label(nd.array(mask).as_in_context(ctx)).asnumpy().astype(bool).astype(np.int8)
                fake_label = nd.array(mask_patch).as_in_context(ctx)
                output = netD(fake_concat)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label,], [output,])
                # Train with real image
                real_concat = nd.concat(real_in, real_out, dim=1)
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label,], [output,])
            trainerD.step(data.shape[0])
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                p5,p6,p7,p8,fake_out = netG(real_in)
                I_comp_1 = nd.array(np.where(mask_b,real_out.asnumpy(),fake_out.asnumpy())).as_in_context(ctx) 
                fake_concat = nd.concat(real_in, fake_out, dim=1)
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1 + L1_loss(real_out*(1-mask), fake_out*(1-mask))*lambda1*6 +L1_loss(real_out_256, p8) * lambda1*0.8 +L1_loss(real_out_128, p7) * lambda1*0.6
                errG.backward()
            trainerG.step(data.shape[0])
            name, acc = metric.get()
            print('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            print('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'%(nd.mean(errG).asscalar(),
                           nd.mean(errG).asscalar(), acc, iter, epoch))
            # print ('L_perceptual = %f, L_style = %f, L_tv = %f, L_total = %f'%(nd.mean(L_perceptual).asscalar(),nd.mean(L_style).asscalar(), nd.mean(L_tv).asscalar(),nd.mean(L_total).asscalar()))
            if (epoch+1)% 50 ==0:
                netG.collect_params().save (args.checkpoint +'/net_%d.params'%(epoch))
            ############################
            # (3) cal vgg16: style_loss+perprocess_loss+tv_loss
            ###########################
            with autograd.record():
                _,_,_,_,fake_out = netG(real_in)
                I_comp = nd.array(np.where(mask_b,real_out.asnumpy(),fake_out.asnumpy())).as_in_context(ctx)
                hout = extract_features(fake_out,style_layers,net,ctx)
                hgt = extract_features(real_out,style_layers,net,ctx)
                hcomp = extract_features(I_comp,style_layers,net,ctx)
                L_perceptual = calc_loss_perceptual(hout,hcomp,hgt)
                L_style = cal_loss_style(hout,hcomp,hgt) #Loss style out and comp
                L_tv = tv_loss(fake_out)
                # L_total = 0.5 * L_perceptual + 50.0 * L_style + 25.0 * L_tv + GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1 + L1_loss(real_out*(1-mask), fake_out*(1-mask))*lambda1*6 +L1_loss(real_out_256, p8) * lambda1*0.8 +L1_loss(real_out_128, p7) * lambda1*0.6
                L_total = 0.5 * L_perceptual + 50.0 * L_style + 25.0 * L_tv
                L_total.backward()
            trainerV.step(data.shape[0])
            print ('L_perceptual = %f, L_style = %f, L_tv = %f, L_total = %f'%(nd.mean(L_perceptual).asscalar(),nd.mean(L_style).asscalar(), nd.mean(L_tv).asscalar(),nd.mean(L_total).asscalar()))
            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                         %(nd.mean(errD).asscalar(),
                           nd.mean(errG).asscalar(), acc, iter, epoch))
            iter = iter + 1
            btic = time.time()
        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--img_size', nargs='?', type=int, default=512, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=0.0005, 
                        help='Learning Rate')
    parser.add_argument('--beta', nargs='?', type=float, default=0.0002, 
                        help='beta')
    parser.add_argument('--trainset_path', nargs='?', type=str, default=None,    
                        help='Path to train images')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--model', nargs='?', type=str, default='',    
                        help='Path to  saved model to restart from')
    parser.add_argument('--gpu', nargs='?', type=bool, default=True, 
                        help='use_gpu')
    args = parser.parse_args()

    train(args)
