from typing import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as tf
from torch.utils.data import random_split 
from torchvision.datasets import CIFAR10

from pathlib import Path

import skimage as ski
import skimage.io

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import helper #Pomoćne funkcije definirane u vježbi


DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'/ 'cifar-10-batches-py'
SAVE_DIR = Path(__file__).parent /'cifar10_log'


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


def prepare_data(): 
    img_height = 32
    img_width = 32
    num_channels = 3
    num_classes = 10

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)


    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))


    
    ds_train = [train_x,train_y]
    ds_valid = [valid_x,valid_y]
    ds_test = [test_x,test_y]

    return ds_train, ds_valid, ds_test, [data_mean,data_std]




class myModel(nn.Module):

    def __init__(self):
        super().__init__()
        #W - širina kanala
        #F - Širina jezgre/filtra
        #P - Širina paddinga
        #S - Stride
        #Nakon konvolucije izlazi su tipa: floor( (H - F + 2P)/S + 1 )
        #Nakon maxpooling opearacije izlazi su tipa: floor( (H - F)/S +1 )

        self.net = nn.Sequential( OrderedDict([
            #Ulazne slike su tipa (RGB slika)3x32x32
            
            ('conv1',nn.Conv2d(in_channels=3, out_channels=16,kernel_size=5,padding= 2  )),
            #Nakon prve konvolucije: (32 -5 +2*2 )/1 + 1 = 32x32
            ('relu1', nn.ReLU()),
            ('maxpool1',nn.MaxPool2d(kernel_size=3,stride = (2,2))),
            #Nakon prvog maxpool-a: (32 - 3)/2 + 1 =  15 x 15

            ('conv2',nn.Conv2d(in_channels = 16, out_channels= 32,kernel_size = 5,padding = 2)),
            #Nakon druge konvolucije: (15 - 5 + 2*2)/1 + 1 = 15 x15
            ('relu2',nn.ReLU()),
            ('maxpool2',nn.MaxPool2d(kernel_size = 3,stride=(2,2) )),
            #Nakon drugog maxpool-a: (15 - 3)/2 + 1 = 7x7
            
            #Sve skupa ovdje nas čeka 32x7 x7= 1568 značajki
            ('flatten',nn.Flatten()),
            ('fc1', nn.Linear(in_features= 1568,out_features=256)),
            ('relu3', nn.ReLU()),
            ('fc2',nn.Linear(in_features= 256, out_features= 128)),
            ('relu4',nn.ReLU()),
            ('fc_out', nn.Linear(in_features = 128, out_features= 10))
        ] ) )



    def forward(self,x):

        return self.net(x)

    def classify(self,x):
        
        return np.argmax(self.forward(x).detach().numpy() ,axis = 1)


    def loss_fn(self,x,y_true):
        fwd = self.forward(x)
        logsum = torch.log(torch.sum(torch.exp(fwd), dim =1))
        linsum = torch.sum(fwd*y_true,dim = 1) #Sumiraj po klasama
        return torch.mean(logsum - linsum)



def evaluate(y_true,y_pred):


    CM = confusion_matrix(y_true,y_pred) #Confusion matrix
    cm_diag = np.diag(CM)
    n_classes = CM.shape[0]
    P = np.zeros(n_classes) #Class Precision
    R = np.zeros(n_classes) #Class Recall
    A = np.zeros(n_classes) #Class Accuracy
    F1 = np.zeros(n_classes)# F1 measure
    for i in range(n_classes):
        TP = cm_diag[i]
        FP = np.sum(CM[i,:]) - cm_diag[i]
        FN = np.sum(CM[:,i])- cm_diag[i]
        P[i] = TP/(FP + TP)
        R[i] = TP/(FN + TP)
        A[i] = TP/np.sum(CM[:,i])
        F1[i] = 2*P[i]*R[i]/(P[i] + R[i])


    avgA = np.sum(A)/n_classes #Average accuracy
    
    metrics = {'CM': CM,
               'avgA': avgA,
               'A': A,
               'P':P,
               'R':R,
               'F1':F1 }

    return metrics

def print_performance(train_perf,valid_perf,epoch):
    
    print(f"\nEpoch: {epoch}")
    print("Performance on training set")
    print(f"Average accuracy:\n{train_perf['avgA']}")
    print(f"Class accuracy:\n{train_perf['A']}")
    print(f"Class precision:\n{train_perf['P']}")
    print(f"Class recall:\n{train_perf['R']}")
    print(f"Class F1 measure:\n{train_perf['F1']}")

    print("Performance on validation set")
    print(f"Average accuracy:\n{valid_perf['avgA']}")
    print(f"Class accuracy:\n{valid_perf['A']}")
    print(f"Class precision:\n{valid_perf['P']}")
    print(f"Class recall:\n{valid_perf['R']}")
    print(f"Class F1 measure:\n{valid_perf['F1']}")




def training_loop(model,ds_train,ds_val,n_epochs = 50,batch_size = 50,lr = 1e-1):

    progress = {'train_loss':[],
                'valid_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'lr': []
    }


    train_x,train_y = ds_train
    train_x = torch.tensor(train_x).detach().float()
    train_y = torch.tensor(train_y).detach()

    val_x,val_y = ds_val
    val_x = torch.tensor(val_x).detach().float()
    val_y = torch.tensor(val_y).detach()

    train_y_categorical = torch.argmax(train_y,dim = 1).detach().numpy()
    val_y_categorical = torch.argmax(val_y,dim = 1).detach().numpy()

    optimizer = optim.SGD(model.parameters(),lr = lr)
    sheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.75)
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(train_x))
        shuff_x = torch.clone(train_x).detach().float()[perm]
        shuff_y = torch.clone(train_y).detach().float()[perm]

        x_batches = torch.split(shuff_x,batch_size)
        y_batches = torch.split(shuff_y,batch_size)
        
        model.train()#Postavlja model u mod rada za treniranje
        for x,y in zip(x_batches,y_batches):
            loss = model.loss_fn(x,y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        sheduler.step()
        
        progress['train_loss'].append(loss.detach().numpy())
        progress['lr'].append(sheduler.get_last_lr())
    
        model.eval()#Postavlja model u evaluacijski mod rada
        with torch.no_grad():
            #Evaluation on validation set
            y_pred = model.classify(val_x)
            val_metrics = evaluate(val_y_categorical,y_pred)
            val_loss = model.loss_fn(val_x,val_y).detach().numpy()
            #Evaluation training set
            y_pred = model.classify(train_x)
            train_metrics = evaluate(train_y_categorical,y_pred)

        progress['valid_loss'].append(val_loss)
        progress['train_acc'].append(train_metrics['avgA'])
        progress['valid_acc'].append(val_metrics['avgA'])
        print_performance(train_metrics,val_metrics,epoch+1)
        
    return progress

    


if __name__ == '__main__':

    ds_train,ds_valid,ds_test,[ data_mean, data_std] = prepare_data()
    

    model = myModel()
    helper.draw_conv_filters(1, 0, model.net.conv1.weight.detach().numpy(), SAVE_DIR)
    
    progress_data = training_loop(model,ds_train,ds_valid)
    helper.draw_conv_filters(0, 0, model.net.conv1.weight.detach().numpy(), SAVE_DIR)
    helper.plot_training_progress(SAVE_DIR,progress_data)
    

    data_x,data_y = ds_valid
    losses = []
    for i,(im,lab) in enumerate(zip(data_x,data_y)):
        imag = torch.tensor(im).detach().float().view(1,3,32,32)
        label = torch.tensor(lab).detach()
        losses.append(
        [model.loss_fn(imag,label).detach().numpy(),i]
        )

    losses.sort(key = lambda x:x[0],reverse=True)

    for n in range(20):

        l,i = losses[n]
        print(f"Loss on image:{l}")
        label = np.argmax(data_y[i])
        print(f"Correct label: {label}")
        prediction = model.forward(torch.tensor(data_x[i]).view(1,3,32,32))
        top3 = torch.topk(prediction,3).indices
        print(f"Top 3 label predcitions:{top3}")
        
        helper.draw_image(data_x[i],data_mean,data_std)
