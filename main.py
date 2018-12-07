# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:07:09 2018

@author: Baek
"""
from option import args
import os

from mydata import make_dataloader

from trainer import Trainer


train_loader, test_loader , classes = make_dataloader(args) 
model = Trainer(args, train_loader, test_loader, classes)

model.run()
