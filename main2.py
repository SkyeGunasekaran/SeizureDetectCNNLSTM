import numpy as np
import numpy as np
import json
import os
import os.path
import glob
import random
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import logging
import time
import gc

# torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,TensorDataset

# local imports
from models.ConvLSTM import ConvNN
from utils.prep_data import train_val_loo_split,train_val_test_split
from utils.load_data_FB_kaggle.load_signals import PrepData
from utils.log import log
from myio.save_load import write_out
from utils.early_stoping import EarlyStopping

def find_largest_divide_num(num):
    for i in range(1, 350):
        if num % i == 0:
            #4
            largest_divisor = i
    print("Largest divisor of {} is {}".format(num,largest_divisor))
    return largest_divisor


def normal(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))+1e-6

    return normalized

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass
def print_epoch_AUC(data_loader, SNN_net,data_size, device, data_type,batch_size):

    y_label = np.empty((data_size), dtype=int)
    y_predict = np.empty((data_size), dtype=float)

    test_batch = iter(data_loader)
    i_batch = 0
    # Minibatch training loop
    for data_it, targets_it in test_batch:

        labels = targets_it.to(device,dtype=torch.int)
        images = data_it.to(device,dtype=torch.float)
        
        outputs, test_mem_rec = SNN_net(images.permute(1,0,2,3))
        _, predicted = outputs.sum(dim=0).max(1)

        y_label[i_batch*batch_size:(i_batch+1)*batch_size] = labels.cpu().numpy()
        y_predict[i_batch*batch_size:(i_batch+1)*batch_size] = predicted.cpu().numpy()
        i_batch +=1
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_predict) 

    result = metrics.auc(fpr, tpr)
    print(data_type+' AUC: ', result)
    return result

def avg_loss(train_losses, test_losses, avg_train_losses, avg_test_losses):
    train_loss = np.average(train_losses)
    test_loss = np.average(test_losses)

    avg_train_losses.append(train_loss)
    avg_test_losses.append(test_loss)

    # scheduler.step()
    train_losses = []
    test_losses = []

    return avg_train_losses, avg_test_losses, train_loss, test_loss
def main(dataset, build_type):
    print ('Main')
    if args.dataset == 'CHBMIT':
        with open('SETTINGS_%s.json' %args.dataset) as f:
            settings = json.load(f)
     # skip Patient 12, not able to read

    targets = [
            '1',
            #'2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            #'12',
            #'13',
            '14',
            '15',
            #'16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23'
    ]
    targets = ['7','21','22','23']
    summary = {}
    epochs = 40
    batch_size = 32

    for target in targets:
        ckpt_target = os.path.join(settings["resultdir"],target)
        makedirs(ckpt_target)
        ictal_X, ictal_y = PrepData(target, type='ictal', settings=settings).apply()
        interictal_X, interictal_y = PrepData(target, type='interictal', settings=settings).apply()
        
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)
        model = ConvNN(X_train.shape).to("cuda")
        fn_weights = "weights_%s_%s.h5" %(target, build_type)
        if os.path.exists(fn_weights):
            model.load_trained_weights(fn_weights)
        else:
            model.fit(batch_size=32,epochs=100,mode=build_type, target = target, X_train = X_train, Y_train = y_train)
            model.evaluate(X_test, y_test)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--dataset", help="FB, CHBMIT or Kaggle2014Pred")
    args = parser.parse_args()
    assert args.mode in ['cv','test']
    main(dataset=args.dataset, build_type=args.mode)

