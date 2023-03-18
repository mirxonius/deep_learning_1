from collections import OrderedDict
from pathlib import Path
from random import shuffle
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as tf
from torch.utils.data import random_split 
from torchvision.datasets import MNIST

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'my_out'


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]



def prep_data():

    ds_train, ds_test = MNIST(DATA_DIR,train= True,download=True), MNIST(DATA_DIR,train=False,download=True)

    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    train_y = ds_train.targets.numpy()

    #perm = np.random.permutation(len(train_x))
    #train_x = train_x[perm]
    #train_y = train_y[perm]
    
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

    train_data = [train_x,train_y]
    valid_data = [valid_x,valid_y]
    test_data = [test_x,test_y]


    return train_data,valid_data,test_data


class myConvModel(nn.Module):

    def __init__(self):
        super().__init__()

        #W - Channel width
        #F - Filter width
        #P - Paddng
        #S - Stride
        #Convolution output shape is: floor( (W +2P - F )/S +1)



        self.net = nn.Sequential(OrderedDict([
        #Input 28x28
        ('conv1',nn.Conv2d(in_channels=1, out_channels=16,kernel_size=5,padding=2,stride = 1) ),
        #After the first conv: (28 - 5 + 4)/1 + 1 = 28
        ('maxpool1', nn.MaxPool2d(kernel_size = 2,stride=2)),
        #First MaxPool: 14x14
        ("relu1",nn.ReLU()),
        ("conv2",nn.Conv2d(in_channels = 16, out_channels=32,kernel_size=5,padding=2)),
        #Second convolution: (14 -5 + 4)/1 +1 = 14
        ("maxpool2",nn.MaxPool2d(kernel_size=2,stride=2)),
        #Second MaxPool: 7x7 
        ("relu2",nn.ReLU()),
        #Altoghether we are left with 32 channels of 7x7 images -> 32*7*7 is the shape of the FFN head
        ("flatten", nn.Flatten()),
        ('fc_1', nn.Linear(in_features=32*7*7,out_features= 512)),
        ('relu3', nn.ReLU()),
        ('fc_2',nn.Linear(in_features=512,out_features=10))
        ])    
        )

    def forward(self,x):
        return self.net(x)

    def classify(self,x):
        return torch.argmax(self.forward(x),dim = 1)


    def get_weights(self):
        """
        Returns:
        List of weights for every layer,
        biases not included
        """
        weights = [
        self.net.conv1.weight,
        self.net.conv2.weight,
        self.net.fc_1.weight,
        self.net.fc_2.weight]

        return weights


    def L2norm(self):
        """
        This function in needed becuase the typical weight_decay
        parameter on SGD penalises both weights and biases.
        We want to penalize only the weights
        Returns:
        Square of the L2 norm of model weights
        """
        weights = self.get_weights()
        norm = 0
        for w in weights:
            norm += torch.norm(w)

        return norm

    
def loss_fn(model, X, Y,weight_decay = 0 ):
    """Arguments:
    model: nn.Module subclass
    X: data images
    Y: data labels
    weight_decay: regularziation parameter
    
    Returns:
    L: Cross Entropy loss + Regularization loss
    """
    fwd = model.forward(X)
    logsum = torch.log(torch.sum(torch.exp(fwd), dim =1))
    linsum = torch.sum(fwd*Y,dim = 1) #Sumiraj po klasama
    
    regL = 0.5*weight_decay*model.L2norm()


    return torch.mean(logsum - linsum) + regL


def evaluate_model(val_x,val_y):
    """Calcualates performace metrics
        Arguments:
        model: Model to be evaluated
        val_x: validation data
        val_y: validation targets

        Returns: Tuple
         A: macro accuracy 
         P: macro precision 
         R: macro recall 
         F1: macro F1 measure
         avgP: Average precision
    """

    y_pred = model.classify(val_x).detach().numpy()
    y_true = np.argmax(val_y,axis = 1)
    CM = confusion_matrix(y_true,y_pred)
    cm_diag = np.diag(CM)
    n_classes = CM.shape[0]
    P = np.zeros(n_classes)
    R = np.zeros(n_classes)
    A = np.zeros(n_classes)
    F1 = np.zeros(n_classes)
    for i in range(n_classes):
        TP = cm_diag[i]
        FP = np.sum(CM[i,:]) - cm_diag[i]
        FN = np.sum(CM[:,i])- cm_diag[i]
        P[i] = TP/(FP + TP)
        R[i] = TP/(FN + TP)
        A[i] = TP/np.sum(CM[:,i])
        F1[i] = 2*P[i]*R[i]/(P[i] + R[i])

    avgA = np.sum(A)/n_classes
    avgR = np. sum(R)/n_classes
    F1 = np.sum(F1)/n_classes
    avgP = np.sum(P)/n_classes

    return avgA,avgR,avgP,F1



def training_loop(model,train_data,val_data,num_epochs = 8,batch_size = 50,learning_rate = 1e-1, weight_decay = 1e-3,save_dir=None):
    """Arguments:
        model: instance of myConvModel to be trained
        X: image training data
        Y: image labels
        num_epochs: number of training epochs
        batch_size: number of images per batch
        lr: learning rate
        weight_decay: L2 regularization factor
    Returns: 
    training_history: loss at the end of every epoch
    """
    training_history = []
    val_history = []

    train_x, train_y = train_data
    train_loader = DataLoader(train_data)
    train_x = torch.tensor(train_x).detach().float()
    train_y = torch.tensor(train_y).detach()
    
    val_x,val_y = val_data
    val_x = torch.tensor(val_x).detach().float()
    val_y = torch.tensor(val_y).detach()

    optimizer = optim.SGD(model.parameters(),lr = learning_rate)


    for n in range(num_epochs):
        perm = torch.randperm(len(train_x))
        #shuff_x = torch.clone(train_x).detach().float()[perm]
        #shuff_y = torch.clone(train_y).detach().float()[perm]

        #x_batches = torch.split(shuff_x,batch_size)
        #y_batches = torch.split(shuff_y,batch_size)

        for x,y in tqdm(train_loader):

            loss = loss_fn(model,x, y,weight_decay=weight_decay)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        training_history.append(loss.detach().numpy())
        
        with torch.no_grad():
            val_loss = loss_fn(model,val_x,val_y,weight_decay=weight_decay)
            val_history.append(val_loss.detach().numpy())
            Acc,Rec,Prec,F1 = evaluate_model(model,val_x,val_y)

        print(f"\nEpoch {n +1} of {num_epochs}")
        print("Performance on validation set:")
        print(f"Average accuracy: {Acc}")
        print(f"Average recall: {Rec}")
        print(f"Average precision: {Prec}")
        print(f"Average F1 measure: {F1}")
        if save_dir != None:
            draw_conv_filters(model,n+1,save_dir)




    return training_history, val_history


def draw_conv_filters(model,epoch,save_dir):
    weights = model.net.conv1.weight
    C ,idle, W,H= weights.shape
    C_half = C//2
    frame,fig = plt.subplots(2,C_half)
    
    for i in range(2):
        for j in range(C_half):
            fig[i,j].imshow(weights[C_half*i +j,0].detach().numpy())

            fig[i,j].tick_params(
                axis='both',         
                which='both',      
                bottom=False,      
                top=False,
                left = False,
                right = False,         
                labelbottom=False,
                labelleft=False 
                ) 
    
    plt.savefig(os.path.join(save_dir,f"covn1_weights_epoch_{epoch}.png" ))
    plt.close()


if __name__ == '__main__':
    
    
    train_data,valid_data,test_data = prep_data()
    
    lambdas = [1e-3,1e-2,1e-1]
    
    file_paths = [SAVE_DIR/f"lambda_{lam}" for lam in lambdas]
    
    for i, lam in enumerate(lambdas):
        model = myConvModel()
        thistory, vhistory = training_loop(model,train_data,valid_data,weight_decay=lam, save_dir = file_paths[i] ) 

        f = open(file_paths[i]/"progress.txt","a")
        for n in range(len(thistory)):
            f.write(f"\nEpoch {n +1}\t Train loss: {thistory[n]}\t Validation loss: {vhistory[n]}")
        f.close()
        
        frame, fig = plt.subplots(figsize = (8,8))
        fig.plot( range(1,len(thistory)+1 ), thistory,label = "training error")
        fig.plot( range(1,len(vhistory)+1 ), vhistory,label = "validation error")
        fig.scatter( range(1,len(thistory)+1 ), thistory)
        fig.scatter( range(1,len(vhistory)+1 ), vhistory)
        fig.set_title(r"$\lambda$ = {}".format(1e-3))
        fig.legend(fontsize = 15)
        plt.savefig(file_paths[i]/"Loss_fn")