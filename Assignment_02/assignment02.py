# You are allowed to use all the modules and packages of pytorch, numpy

import torch.nn as nn
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.misc
import imageio
import os
import numpy as np
from AutoencoderNet import AutoencoderNet
from torch.autograd import Variable
from math import log10

# Hyper Parameters
num_epochs = 10
batch_size = 100


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def TrainAutoencoder(train_loader, test_loader, num_epochs):
    # Trains autoencoder and returns the trained model.
    # The trained model will be used as feature extractor to train linear classifier.

    lr = 1e-3
    model = AutoencoderNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # TODO : define ans set the optimizer

    criterion = nn.CrossEntropyLoss()  # TODO : define the loss function

    for epoch in range(num_epochs):

        model.train()
        lr = lr * (0.1 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training loop for an epoch
        for i, (images, _) in enumerate(train_loader):

            # TODO : implement a full step of optimization.
            # Convert torch tensor to Variable
            images = Variable(images)
            labels = images

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Autoencoder Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss))

        model.eval()
        pred_imgs = np.zeros((100 * 28, 200 * 28), dtype=np.uint8) + 100
        avg_psnr = 0
        # Testing loop for an epoch
        for images, labels in test_loader:
            # TODO : calculate the outputs

            mse = torch.mean((outputs - input_var.cpu()).pow(2))
            psnr = 10 * log10(1 / mse)
            avg_psnr += psnr

        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))

    return model

'''
def TrainLinearClassifier(model, train_loader, test_loader, num_epochs):
    lr = 1e-3
    model.eval()
    # Get the feature dimension by passing a random input
    feat_dim = model.get_features(torch.autograd.Variable(torch.FloatTensor(1, 1, 28, 28))).shape[1]

    lin_criterion =  # TODO : define the loss
    lin_model = torch.nn.Sequential(torch.nn.Linear(feat_dim, 10))
    lin_optimizer =  # TODO : define the optimizer

    top1 = AverageMeter()
    top5 = AverageMeter()

    for epoch in range(num_epochs):

        lin_model.train()
        top1.reset()
        top5.reset()
        lr = lr * (0.1 ** (epoch // 3))
        for param_group in lin_optimizer.param_groups:
            param_group['lr'] = lr

        # training for loop
        for i, (images, target) in enumerate(train_loader):

            # target includes the character labels
            # get the encoder features
            feats = model.get_features(torch.autograd.Variable(images))

            # TODO : implement a full step of optimization.

            prec1, prec5 = accuracy(pred.data, target, topk=(1, 5))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            if i % 100 == 0:
                print('Linear classifier Epoch: [{0}][{1}/{2}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), top1=top1, top5=top5))

        lin_model.eval()
        top1.reset()
        top5.reset()
        for i, (images, target) in enumerate(test_loader):
            # TODO : extract the feature and compute classifier predictions

            prec1, prec5 = accuracy(pred.data, target, topk=(1, 5))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

        print('===>>>\t'
              'Prec@1 ({top1.avg:.3f})\t'
              'Prec@5 ({top5.avg:.3f})'.format(top1=top1, top5=top5))
'''

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = TrainAutoencoder(train_loader, test_loader, num_epochs)
#TrainLinearClassifier(model, train_loader, test_loader, num_epochs)
