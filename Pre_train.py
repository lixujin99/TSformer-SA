#!/usr/bin/env python
# coding: utf-8
# In[1]:
#GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import glob
import time
import math
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import collections
import random
from tqdm import tqdm
from torch import nn
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import scipy
from config_tff import Config
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import copy
from load_data import load_data_pre
from model import TSformer_SA
import pywt
from loss import ContrastiveLoss
import gc
import sys


# In[75]:
dist.init_process_group(backend="nccl")#, init_method="env://", world_size=torch.cuda.device_count(),rank=local_rank)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

config = Config()


# In[76]:

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    al = labels.shape[0]
    TP = 0   
    FP = 0 
    TN = 0   
    FN = 0  
    for i in range(al):
        if ((preds[i]==1)and(labels[i]==1)):
            TP += 1
        if ((preds[i]==1)and(labels[i]==0)):
            FP += 1
        if ((preds[i]==0)and(labels[i]==1)):
            FN += 1
        if ((preds[i]==0)and(labels[i]==0)):
            TN +=1
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc1 = TP/(TP+FN)
    acc2 = TN/(TN+FP)
    BN = (acc1+acc2)/2
    return correct / len(labels),acc1,acc2,BN
    
def acc10(output, labels):
    preds = output.max(1)[1].type_as(labels)
    al = labels.shape[0]
    TP = 0   
    TN = 0   
    num_1 = 0
    for i in range(al):
        if ((preds[i]==1)and(labels[i]==1)):
            TP += 1
        if ((preds[i]==0)and(labels[i]==0)):
            TN += 1
        if (labels[i]==1):
            num_1 += 1
    return num_1,TP,TN
    

def z_score_channel(x):
    x = x.permute(0,2,1,3)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j, :, :] = (x[i, j, :, :] - x[i, j, :, :].mean()) / x[i, j, :, :].std()
    x = x.permute(0,2,1,3)
    return x


def Wave(Raw):
    Raw = Raw.cpu()
    Raw = Raw.data.numpy()
    Raw = np.float16(np.squeeze(Raw))

    wavename = config.wavename
    totalscal = Raw.shape[-1]//2  
    sr = Raw.shape[-1]
    Fc = pywt.central_frequency(wavename)               
    cparam = 2*Fc*totalscal      
    scales = cparam / np.arange(totalscal, 1, -1) 
    scales = scales[scales.shape[0]-config.scale:]

    wav_matrix, _ = pywt.cwt(Raw, scales, wavename, 1/sr)
    wav_matrix = abs(wav_matrix)
    return wav_matrix
    

class MyDataset(Dataset):
    def __init__(self, data_eeg, label_eeg):
                                                             
        self.data_eeg = data_eeg
        self.label_eeg = label_eeg

        self.data_wave = []
        for i_wave in range(data_eeg.shape[0]):
            self.data_wave.append(Wave(data_eeg[i_wave]))
        self.data_wave = np.array(self.data_wave)
        self.data_wave = torch.FloatTensor(self.data_wave)
        #self.data_wave = z_score_channel(self.data_wave)
        if (dist.get_rank() == 0):
            print(self.data_wave.shape)

    def __getitem__(self, index):
        eeg = self.data_eeg[index]
        eeg_wave = self.data_wave[index]
        eeg_label = self.label_eeg[index]

        return eeg, eeg_wave, eeg_label

    def __len__(self):
        return self.data_eeg.size(0)
    
class MyDataset_test(Dataset):
    def __init__(self, data_eeg, label_eeg):
                                                               
        self.data_eeg = data_eeg
        self.label_eeg = label_eeg

    def __getitem__(self, index):
        eeg = self.data_eeg[index]
        eeg_label = self.label_eeg[index]

        eeg_wave = Wave(eeg)
        eeg_wave = torch.FloatTensor(eeg_wave)

        return eeg, eeg_wave, eeg_label

    def __len__(self):
        return self.data_eeg.size(0)


# In[80]:


# In[7]:
#导入数据
if config.train == 'people':
    name = config.name_people
    path1 = config.path1_people
elif config.train == 'car':
    name = config.name_car
    path1 = config.path1_car
elif config.train == 'plane':
    name = config.name_plane
    path1 = config.path1_plane
path2 = config.path2

epochs = config.epoch

BNmatrix = np.zeros(len(name))
AUCmatrix = np.zeros(len(name))
F1matrix = np.zeros(len(name))
PRmatrix = np.zeros((len(name),2))
Result = np.zeros((5,2))

for id_name in range(len(name)):
    name_test = [name[id_name]]
    name_train = copy.deepcopy(name)
    del name_train[id_name]
    
    name1 = name_train
    model = TSformer_SA()

    model = model.to(device) 
    if (torch.cuda.device_count() > 1)and(dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_con = ContrastiveLoss(config.temperature).to(device)

    t_total = time.time()
    
    if (dist.get_rank() == 0):
        print(name_test[0])

    save_model = config.save + name_test[0] + 'pre_train.pkl'
    
    train_data, train_label, val_data, val_label = load_data_pre(path1,path2,name1)
    if (dist.get_rank() == 0):
        print(train_data.shape)
    EEG_dataset1 = MyDataset(train_data,train_label)
    EEG_dataset2 = MyDataset(val_data,val_label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(EEG_dataset1)
    nw = 2
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=config.batchsize,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=2, shuffle=False)

    dist.barrier()
    del EEG_dataset1, EEG_dataset2
    gc.collect()
    
    ############################################################################ Pre-training ########################################################################################
    step = config.lr_mae
    val_max = 0
    stepp_new = 0
    
    for i in range(epochs):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%40 == 0 and i>0):
            step = step*0.8
        optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=0.01)
        train_l_sum, train_s_sum, train_fc_sum, train_ts_sum, train_fs_sum, train_c_sum, train_con_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for ii1, data in enumerate(trainloader, 0):
            inputs, waves, labels = data
            optimizer.zero_grad()
            inputs = inputs.to(device)
            waves = waves.to(device)
            labels = labels.to(device)
            outputs_label, t_con, f_con = model(inputs,waves)
            loss_c = criterion(outputs_label, labels.long())
            loss_con = criterion_con(t_con,f_con)
            loss = loss_c + loss_con
            loss.backward()
            optimizer.step()

            train_acc_sum += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(outputs_label, labels)
            acc1_sum += num_acc1
            acc0_sum += num_acc0
            sum_1 += num_1
            n += labels.shape[0]
            
            train_l_sum += loss.cpu().item()
            train_c_sum += loss_c.cpu().item()
            train_con_sum += loss_con.cpu().item()
            
        train_l_sum = train_l_sum / (ii1+1)
        train_c_sum = train_c_sum / (ii1+1)
        train_con_sum = train_con_sum / (ii1+1)
        sum_0 = n - sum_1
        BN = train_acc_sum / n
        acc1 = acc1_sum/sum_1
        acc0 = acc0_sum/sum_0 

        #Validation
        val_l_sum, val_t_sum, val_f_sum, val_ts_sum, val_fs_sum, val_c_sum, val_con_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0
        for ii2, data in enumerate(valloader, 0):
            val_inputs, val_waves, val_labels = data
            with torch.no_grad():
                val_inputs = val_inputs.to(device)
                val_waves = val_waves.to(device)
                val_labels = val_labels.to(device)
                val_output, val_tcon, val_fcon = model(val_inputs,val_waves)
                loss_val_c = criterion(val_output, val_labels.long())
                loss_val = loss_val_c

            
            val_acc_sum += (val_output.argmax(dim=1) == val_labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(val_output, val_labels)
            val_acc1_sum += num_acc1
            val_acc0_sum += num_acc0
            sum_1 += num_1
            n += val_labels.shape[0]

            val_l_sum += loss_val.cpu().item()
            val_c_sum += loss_val_c.cpu().item()

        val_l_sum = val_l_sum / (ii2+1)
        val_c_sum = val_c_sum / (ii2+1)
        sum_0 = n - sum_1
        val_acc1 = val_acc1_sum/sum_1
        val_acc0 = val_acc0_sum/sum_0
        val_BN = (val_acc1 + val_acc0) / 2

        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                  'loss_cls: {:.4f}'.format(train_c_sum),
                  'loss_con: {:.4f}'.format(train_con_sum),
                  'acc1= {:.4f}'.format(acc1),
                  'acc0= {:.4f}'.format(acc0),
                  'BN= {:.4f}'.format(BN),
                  'loss_val: {:.4f}'.format(val_l_sum),
                  'loss_cls_val: {:.4f}'.format(val_c_sum),
                  "val_acc1= {:.4f}".format(val_acc1),
                  "val_acc0= {:.4f}".format(val_acc0),
                  "val_BN= {:.4f}".format(val_BN),
                  "time: {:.4f}s".format(time.time() - t))
        
        if (val_max<val_BN):
            val_max = val_BN
            if (dist.get_rank() == 0):
                torch.save(model.state_dict(), save_model)
        
    dist.barrier()
    if (dist.get_rank() == 0):
        print('Finished Training')
        model.load_state_dict(torch.load(save_model,map_location=device))



#python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 /TSformer-SA/Pre_train.py