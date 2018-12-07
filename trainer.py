# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:38:36 2018

@author: Baek
"""

from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model import baseline, resnet20, resnet32, resnet56
"""
"""
from torchvision.utils import save_image
import os


import numpy as np
import matplotlib.pyplot as plt
import cv2
from utility import timer
from warm_multi_step_lr import WarmMultiStepLR
from tqdm import tqdm
import time

def print_save(line, text_path):
    print(line)
    f = open(text_path, 'a')
    f.write(line + '\n')
    f.close()
class Trainer(object):
    def __init__(self, args, training_loader, testing_loader, classes):
        super(Trainer, self).__init__()
        self.args = args
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = args.lr
        self.epochs = args.epochs
        self.seed = args.seed
        self.data_timer = None
        self.train_timer = None
        self.test_timer = None
        self.epoch = 0
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.result = '../' + args.model_path
        self.image_path = self.result + '/image'
        self.loss_type = args.loss_type
        self.text_path = self.result + "/log/lr_{}_bat_{}.txt".format(self.lr, self.args.batch_size)
        if not os.path.exists(self.result + '/log'):
             os.makedirs(self.result + '/log')
        if not os.path.exists(self.result + '/image'):
             os.makedirs(self.result + '/image')
        
    def build_model(self):
        if self.args.mean_teacher:
            self.model = baseline(self.args).to(self.device)
            self.model.weight_init()
            self.ema_model = baseline(self.args).to(self.device)
            self.ema_model.weight_init()
            
        if self.args.pretrain:
            print('Loading  ', self.args.pretrain)
            self.model = torch.load(self.args.pretrain, map_location=lambda storage, loc: storage).to(self.device)
            checkpoint = torch.load(self.args.pretrain)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
        else:
            self.model = baseline(self.args).to(self.device)
            self.model.weight_init()
            #self.model = resnet56().to(self.device)
            if self.args.optim is 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        milestones = []
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=self.args.gamma) 

        #if self.args.mean_teacher:
            #self.model = resnet20_cifar().to(self.device)
            #elf.ema_model = resnet20_cifar().to(self.device)
        print(self.model)
        self.data_timer = timer(self.args)
        self.train_timer = timer(self.args)
        self.test_timer = timer(self.args)

        self.L1 = nn.L1Loss()
        self.L2= nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.L1.cuda()
            self.L2.cuda()
            self.criterion.cuda()
        
         # lr decay
        
        #self.scheduler = WarmMultiStepLR(self.optimizer, milestones=milestones, gamma=self.args.gamma, scale = 10)


    def train(self):
        self.model.train()
        if self.args.mean_teacher:
            self.ema_model.train()
        train_loss = 0
        train_length = 0
        total = 0
        correct = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        class_loss = AverageMeter()
        cons_loss = AverageMeter()
        end = time.time()
        """
        dataloader_iterator = iter(self.training_loader)
        iterations = len(self.training_loader)
        #for batch_num in tqdm(range(len(self.training_loader)), ncols=80):
        for batch_num in range(iterations):
            try:
                inputs, labels = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(self.training_loader)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)
            if self.args.semi_supervised:
                input1 = inputs[0:self.args.labeled_batch_size]
                label1 = labels[0:self.args.labeled_batch_size]
                input2 = inputs[self.args.labeled_batch_size:]
                label2 = labels[self.args.labeled_batch_size:]
                #print(label1, label2)

                out1 = self.model(input1)
                loss = 0
                class_loss = F.cross_entropy(out1, label1) * label1.size(0) / batch_size
                class_loss.update(class_loss)
                loss += class_loss
                    #if self.args.mean_teacher and label2.size(0) > 0:
                weight = self._get_current_consistency_weight()
                if self.args.mean_teacher and weight > 0:
                    out2 = self.model(input2)
                    #ema_out = self.model(input2)
                    ema_out = self.ema_model(input2)
                    ema_out = Variable(ema_out.detach().data, requires_grad=False)
                    out2_softmax = F.softmax(out2, dim=1)
                    ema_out_softmax = F.softmax(ema_out, dim=1)
                    #ema_loss = F.mse_loss(out2_softmax, ema_out_softmax) / 10
                    ema_loss = torch.sum((ema_out_softmax - out2_softmax) ** 2) / (10 * batch_size)
                    if self.args.sntg:
                        loss1, loss2 = SNTG(out2, ema_out)
                        lamda1, lamda2 = 1.0, 1.0
                        ema_loss = lamda1 * loss1 + lamda2 * loss2
                    
                    loss += weight * ema_loss
                    cons_loss.update(weight * ema_loss.item())
                    out = torch.cat((out1, out2), dim=0)
                else:
                    inputs = input1
                    out = out1
                    labels = label1
            else:
                out = self.model(inputs)
                loss = F.cross_entropy(out, labels)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            
            if self.args.mean_teacher:
                self._update_ema_variables(self.epoch)
                
            #print(out.size(), labels.size(), inputs.size())
            prec1 = accuracy(out.data, labels)[0]
            #print(prec1)
            #input()
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            train_length += batch_size
            if (batch_num % self.args.print_every == 0) or (batch_num == len(self.training_loader)):
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {class_loss.val:.4f} ({class_loss.avg:.4f})\t'
                  'Cons Loss {cons_loss.val:.4f} ({cons_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      self.epoch, batch_num, len(self.training_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, class_loss=class_loss,cons_loss=cons_loss,top1=top1))
        """
        for batch_num, (data) in enumerate(self.training_loader):
            
            if self.args.labeled == 50000 or self.args.semi_supervised is None:
                inputs, labels = data[0], data[1]
                mask = torch.ones_like(labels)
            else:
                inputs, labels, masks = data[0], data[1], data[2]
                mask = masks.to(self.device)

            batch_size = inputs.size(0)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            data_time.update(time.time() - end)
            if self.args.semi_supervised:
                mask = mask.type(torch.uint8)
                input1 = Variable(inputs[mask,:])
                label1 = labels[mask]
                input2 = Variable(inputs[1-mask,:])
                label2 = labels[1-mask]
                loss = 0
                #print(label1, label2)
                if label1.size(0) > 0:
                    out1 = self.model(input1)
                    class_loss = F.cross_entropy(out1, label1) * label1.size(0) / batch_size
                    class_loss.update(class_loss)
                    loss += class_loss
                if True:
                #if self.args.mean_teacher and label2.size(0) > 0:
                    weight = self._get_current_consistency_weight()
                    out2 = self.model(inputs)
                    #ema_out = self.model(input2)
                    ema_out = self.ema_model(inputs)
                    ema_out = Variable(ema_out.detach().data, requires_grad=False)
                    out2_softmax = F.softmax(out2, dim=1)
                    ema_out_softmax = F.softmax(ema_out, dim=1)
                    #ema_loss = F.mse_loss(out2_softmax, ema_out_softmax) / 10
                    ema_loss = torch.sum((ema_out_softmax - out2_softmax) ** 2) / (10 * batch_size)
                    if self.args.sntg:
                        loss1, loss2 = SNTG(out2, ema_out)
                        lamda1, lamda2 = 1.0, 1.0
                        ema_loss = lamda1 * loss1 + lamda2 * loss2
                    
                    loss += weight * ema_loss
                    cons_loss.update(weight * ema_loss.item())
                    #out = torch.cat((out1, out2), dim=0)
                    out = out2
            else:
                out = self.model(inputs)
                loss = F.cross_entropy(out1, label1)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            
            if self.args.mean_teacher:
                self._update_ema_variables(self.epoch)
                
            prec1 = accuracy(out.data, labels)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            train_length += batch_size
            if (batch_num % self.args.print_every == 0) or (batch_num == len(self.training_loader)):
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {class_loss.val:.4f} ({class_loss.avg:.4f})\t'
                  'Cons Loss {cons_loss.val:.4f} ({cons_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      self.epoch, batch_num, len(self.training_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, class_loss=class_loss,cons_loss=cons_loss,top1=top1))
            

    def test(self):
        self.model.eval()
        if self.args.mean_teacher:
            self.ema_model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            #for batch_num, (inputs, labels) in enumerate(tqdm(self.testing_loader, ncols=80)):
            for batch_num, (inputs, labels) in enumerate(self.testing_loader):
                batch_size = inputs.size(0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                output = self.model(inputs)
                loss = self.criterion(output, labels) / batch_size

                output = output.float()
                loss = loss.float()

                prec1 = accuracy(output.data, labels)[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                
                #if self.args.test_only:
                    #save_image(inputs/self.args.rgb_range, inputs_name)
        
            if (batch_num % self.args.print_every == 0) or (batch_num == len(self.testing_loader)):
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        batch_num, len(self.testing_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

        print('Test: * Prec@1 {top1.avg:.3f}'
            .format(top1=top1))
            

    def save(self):
        model_out_path = self.result + '/model.pth'
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
            }, model_out_path)
        #torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def _get_current_consistency_weight(self):
        #return self.args.consistency * step_rampup(self.epoch, self.args.consistency_thr)
        return self.args.consistency * sigmoid_rampup(self.epoch, self.args.consistency_thr)
        #return self.args.consistency * self._rampup()
    def _rampup(self):
        if epoch < self.args.consistency_thr:
            p = max(0.0, float(self.epoch)) / float()
            p = 1.0 - p
            return math.exp(-p*p*5.0)
        else:
            return 1.0
            
    def _update_ema_variables(self, global_step):
        #alpha = min(1 - 1 / (global_step + 1), self.args.alpha)
        alpha = self.args.alpha
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            #ema_param.data = ema_param.data * alpha + param.data * (1-alpha)
    def plot(self):
        X = np.asarray([x for x in range(self.epoch+1)])
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        ax2 = fig.add_subplot(3, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Training accuracy')
        ax3 = fig.add_subplot(3, 1, 3)
        plt.xlabel('Epoch')
        plt.ylabel('Test     accuracy')
        ax1.plot(X, np.asarray(self.loss))
        ax2.plot(X, np.asarray(self.trainacc))
        ax3.plot(X, np.asarray(self.testacc))
        plt.savefig(os.path.join(self.result + '/log','loss_acc.png'))

    def run(self):
        self.build_model()
        f = open(self.text_path, 'a')
        line = "lr : {}\n".format(self.lr)
        f.write(line)
        f.close()
        self.loss = []
        self.trainacc =[]
        self.testacc = []

        plt.switch_backend('agg')

        for epoch in range(0, self.epochs):
            self.epoch += 1
            print_save("\n===> Epoch {} starts:".format(self.epoch), self.text_path)
            if not self.args.test_only:
                self.train()
                self.scheduler.step(self.epoch)
                self.test()
                #self.plot()
                #self.save() 
                print(self.scheduler.get_lr())
            if self.args.test_only:
                print('Test start')
                self.test()
                break


def step_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0     

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length
        
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def SNTG(h1, h2):
    ## h1 = model's output
    ## h2 = teacher's output
    f1 = F.softmax(h1, dim=1)
    f2 = F.softmax(h2, dim=1)
    _, y1 = torch.max(f1,dim=1)
    _, y2 = torch.max(f2,dim=1)
    batch_size = output1.size(0)
    W = torch.zeros(batch_size, batch_size)
    for i in range(batch_size):
        for j in range(batch_size):
            W[i, j] = y2(i) == y2(j)

    loss1 = torch.sum((f1- f2) ** 2) /  (batch_size)
    loss2 = 0
    for i in range(batch_size):
        for j in range(batch_size):
            norm2 = (torch.sum(h[i,:] - h[j,:]) ** 2)
            if W[i,j] == 1:
                loss2 += norm2
            else:
                loss2 += max(0, norm2)
    loss2 = loss2 / (batch_size * batch_size)

    return loss1, loss2