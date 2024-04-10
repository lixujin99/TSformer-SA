import numpy as np
from config_tff import Config
import torch.distributed as dist
import torch


config = Config()

def load_data_pre(path1, path2, names, ratio=0.9):
    for id_name1 in range(len(names)):
        path = path1+names[id_name1]+path2
        mat = np.load(path)
        data1c = mat['Data1']
        data1c = data1c[:,np.newaxis,:,:config.T]
        label1c = mat['label1']

        Data0 = data1c[label1c==0,:]
        Data1 = data1c[label1c==1,:]
        rd_index = np.random.permutation(Data0.shape[0])
        data_0_downsampled = Data0[rd_index[:Data1.shape[0]],:]

        rd_index = np.random.permutation(Data1.shape[0])
        train_index = rd_index[:int(ratio*Data1.shape[0])]
        val_index = rd_index[int(ratio*Data1.shape[0]):]
        train_data_p = np.concatenate((Data1[train_index],data_0_downsampled[train_index]),axis=0)
        train_label_p = np.concatenate((np.ones(train_index.shape[0]),np.zeros(train_index.shape[0])),axis=0)
        val_data_p = np.concatenate((Data1[val_index],data_0_downsampled[val_index]),axis=0)
        val_label_p = np.concatenate((np.ones(val_index.shape[0]),np.zeros(val_index.shape[0])),axis=0)
        
        if (id_name1 == 0):
            train_datac = train_data_p
            train_labelc = train_label_p
            val_datac = val_data_p
            val_labelc = val_label_p
        else:
            train_datac = np.append(train_datac, train_data_p, axis = 0)
            train_labelc = np.append(train_labelc, train_label_p)
            val_datac = np.append(val_datac, val_data_p, axis = 0)
            val_labelc = np.append(val_labelc, val_label_p)

    return torch.from_numpy(train_datac), torch.from_numpy(train_labelc), torch.from_numpy(val_datac), torch.from_numpy(val_labelc)


def load_data_alls(path1, path2, name, block, ratio=0.9):
    path = path1+name+path2
    mat = np.load(path)
    data1c = mat['Data1']
    data1c = data1c[:,np.newaxis,:,:config.T]
    label1c = mat['label1']

    Data0 = data1c[label1c==0,:]
    Data1 = data1c[label1c==1,:]

    if config.train=='car':
        num0 = Data0.shape[0]//5
        num1 = Data1.shape[0]//5
    else:
        num0 = Data0.shape[0]//10
        num1 = Data1.shape[0]//10

    train_data0 = Data0[:num0*block]
    train_data1 = Data1[:num1*block]
    test_data0 = Data0[num0*block:]
    test_data1 = Data1[num1*block:]
    test_data = np.concatenate((test_data0,test_data1),axis=0)
    test_label = np.concatenate((np.zeros(test_data0.shape[0]),np.ones(test_data1.shape[0])),axis=0)

    rd_index = np.random.permutation(train_data0.shape[0])
    data_0_downsampled = train_data0[rd_index[:train_data1.shape[0]],:]
    rd_index = np.random.permutation(train_data1.shape[0])
    train_index = rd_index[:int(ratio*train_data1.shape[0])]
    val_index = rd_index[int(ratio*train_data1.shape[0]):]
    train_data = np.concatenate((train_data1[train_index],data_0_downsampled[train_index]),axis=0)
    train_label = np.concatenate((np.ones(train_index.shape[0]),np.zeros(train_index.shape[0])),axis=0)
    val_data = np.concatenate((train_data1[val_index],data_0_downsampled[val_index]),axis=0)
    val_label = np.concatenate((np.ones(val_index.shape[0]),np.zeros(val_index.shape[0])),axis=0)
    
    return torch.from_numpy(train_data), torch.from_numpy(train_label), torch.from_numpy(val_data), torch.from_numpy(val_label), torch.from_numpy(test_data), torch.from_numpy(test_label)


