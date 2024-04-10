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
from load_data import load_data_alls
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
    model = TSformer_SA()

    t_total = time.time()
    
    if (dist.get_rank() == 0):
        print(name[id_name])

    model_pre = config.save + name[id_name] + 'pre_train.pkl'
    
    pre_parameters = torch.load(model_pre)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pre_parameters.items():
        name_para = k[7:] 
        new_state_dict[name_para] = v
    pre_parameters = new_state_dict
    
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in pre_parameters.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device) 
    if (torch.cuda.device_count() > 1)and(dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_con = ContrastiveLoss(config.temperature).to(device)

    train_data, train_label, val_data, val_label, test_data, test_label = load_data_alls(path1,path2,name[id_name],config.block,ratio=1)
    if (dist.get_rank() == 0):
        print(train_data.shape)
    EEG_dataset1 = MyDataset(train_data,train_label)
    EEG_dataset2 = MyDataset(val_data,val_label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(EEG_dataset1)
    nw = 2#min([os.cpu_count(), config.batchsize if config.batchsize > 1 else 0, 8])
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=config.batchsize,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=2, shuffle=False)

    
    ############################################################ Fine-tuning ###############################################################################
    save_model = config.save + name[id_name] + str(config.block) + 'fine_ture.pkl'
    step = config.lr_mae
    val_max = 0
    stepp_new = 0
    
    for i in range(epochs):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%40 == 0 and i>0):
            step = step*0.8
        
        adapter_layers = list(map(id, model.adapter.parameters()))
        base_layers = (p for p in model.parameters() if id(p) not in adapter_layers)
        parameters = [{'params': model.adapter.parameters(), 'lr': step}, {'params': base_layers, 'lr': 0}]
        optimizer = optim.Adam(parameters, weight_decay=0.01)
 
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

        
        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                  'loss_cls: {:.4f}'.format(train_c_sum),
                  'loss_con: {:.4f}'.format(train_con_sum),
                  'acc1= {:.4f}'.format(acc1),
                  'acc0= {:.4f}'.format(acc0),
                  'BN= {:.4f}'.format(BN),
                  "time: {:.4f}s".format(time.time() - t))

        if (val_max<BN):
            val_max = BN
            if (dist.get_rank() == 0):
                torch.save(model.state_dict(), save_model)
        
    dist.barrier()
    if (dist.get_rank() == 0):
        print('Finished Training')
        model.load_state_dict(torch.load(save_model,map_location=device))
    

    # Testing
    EEG_dataset3 = MyDataset_test(test_data,test_label)
    testloader = torch.utils.data.DataLoader(EEG_dataset3, batch_size=256, shuffle=True)

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    Label = None
    pre = None
    pre_prob = None
    loss_t_test = 0
    loss_f_test = 0
    loss_ts_test = 0
    for j, data in enumerate(testloader, 0):
        inputs, waves, labels = data
        inputs = inputs.to(device)
        waves = waves.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs_label, _, _ = model(inputs,waves)

        preds = outputs_label.max(1)[1].type_as(labels)
        Label = labels if Label is None else torch.cat((Label,labels))
        pre = preds if pre is None else torch.cat((pre,preds))
        pre_prob = F.softmax(outputs_label,dim=-1) if pre_prob is None else torch.cat((pre_prob,F.softmax(outputs_label,dim=-1)))
        al = labels.shape[0]
        TP = 0  
        FP = 0.001   
        TN = 0  
        FN = 0.001  
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
        acc_test = correct / len(labels)
        TP_all = TP+TP_all
        TN_all = TN+TN_all
        FP_all = FP+FP_all
        FN_all = FN+FN_all
    
    acc1 = TP_all/(TP_all+FN_all)
    acc0 = TN_all/(TN_all+FP_all)
    BN_all = (acc1+acc0)/2
    BNmatrix[id_name] = BN_all
    AUCmatrix[id_name] = roc_auc_score(Label.cpu().detach().numpy(),pre_prob[:,1].cpu().detach().numpy())
    F1matrix[id_name] = f1_score(Label.cpu().detach().numpy(),pre.cpu().detach().numpy())
    PRmatrix[id_name,0] = acc1
    PRmatrix[id_name,1] = acc0
    loss_t_test = loss_t_test / (j+1)
    loss_f_test = loss_f_test / (j+1)
    loss_ts_test = loss_ts_test / (j+1)

    if (dist.get_rank() == 0):
        print(name[id_name]," Test set results:","acc1= {:.4f}".format(acc1),"acc0= {:.4f}".format(acc0),"BN= {:.4f}".format(BN_all),
                "F1-score= {:.4f}".format(f1_score(Label.cpu().detach().numpy(),pre.cpu().detach().numpy())),"AUC= {:.4f}".format(roc_auc_score(Label.cpu().detach().numpy(),pre_prob[:,1].cpu().detach().numpy())))
    dist.barrier()


BNmatrix = BNmatrix*100
acc = np.mean(BNmatrix)
var = np.var(BNmatrix)
auc_acc = np.mean(AUCmatrix)
auc_std = np.sqrt(np.var(AUCmatrix))
f1_acc = np.mean(F1matrix)
f1_std = np.sqrt(np.var(F1matrix))
std = np.sqrt(var)
std = std

Result[0,0] = acc
Result[0,1] = std
PRmatrix1 = PRmatrix
PRmatrix1[:,1] = 1 - PRmatrix1[:,1]
Result[1:3,0] = np.mean(PRmatrix1*100, axis=0)
Result[1:3,1] = np.sqrt(np.var(PRmatrix1*100, axis=0))
Result[3,0] = f1_acc
Result[3,1] = f1_std
Result[4,0] = auc_acc
Result[4,1] = auc_std

if (dist.get_rank() == 0):
    print(F1matrix)
    print(f1_acc, "+-", f1_std)
    print(AUCmatrix)
    print(auc_acc, "+-", auc_std)
    print(BNmatrix)
    print(PRmatrix*100)
    print(acc, "+-", std)
    print(np.mean(PRmatrix*100, axis=0))

#python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 /TSformer-SA/Fine_tune.py