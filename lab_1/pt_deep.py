from asyncio import protocols
from distutils.command.config import config
from pickletools import read_uint1
import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import data


class PTDeep(nn.Module):

    def __init__(self,configuration_list, activation = None):
        super().__init__()

        W_list = []
        b_list = []
        for i in range(len(configuration_list)-1):
            
            W_list.append(
                nn.Parameter(
                torch.randn(configuration_list[i],configuration_list[i+1],dtype=torch.float),
                requires_grad=True)
            )

            b_list.append(
                nn.Parameter(
                torch.zeros(configuration_list[i+1]),
                requires_grad=True)
            )

        self.weights = nn.ParameterList(W_list)
        self.biases = nn.ParameterList(b_list)
        self.Nw = len(self.weights)

        if activation is None:
            self.activaion = lambda x: x
        else:
            self.activaion = activation


    def count_params(self):
        """Prints names and dimensions of model parameters
        as well as the total number of parameters
        Returns: Total parameters
        
        """
        
        name_and_dim = []

        for param in self.named_parameters():
            name_and_dim.append(
                (param[0],param[1].shape)
            )
        
        for name,dim in name_and_dim:
            print("Parameter name ", name, " Parameter dimension: ",dim)
        
        total_params = np.sum([param.numel() for param in self.parameters()])
        print("Total number of parameters: ",total_params)
        return total_params


    def forward(self, x):
        fwd = x
        for i in range(self.Nw-1):
            fwd = self.activaion(
                torch.mm(fwd,self.weights[i]) + self.biases[i]
                )
        
        #Aktivacija zadnjeg sloja je softmax
        fwd = torch.mm(fwd,self.weights[-1]) + self.biases[-1]
        #Stabilizacija softmaxa
        fwd = fwd - torch.max(fwd,dim = 1).values.view(-1,1).detach()
        return torch.softmax(fwd,dim = 1) 


    def classify(self,x):
        return self.forward(x).argmax(dim = 1).detach().numpy()


    def L2_loss(self,param_lambda):
        L = 0
        for i in range(len(self.weights)):
            L += param_lambda*torch.sum(
                torch.mm(self.weights[i],self.weights[i].T)
                )

        return L

    def get_loss(self,X,Yoh,param_lambda=0):
        #Stabilizirani gubitak
        fwd = self.forward(X)
        
        #linsum = torch.sum(fwd *Yoh,dim = 1)
        #logsum = torch.log(torch.sum( torch.exp(fwd),dim = 1))

        loss = -torch.log( fwd + 1e-15)*Yoh
        loss = torch.sum(loss,dim = 1)


        L2 = 0
        if param_lambda >0:
            L2 = self.L2_loss(param_lambda)
        
        return  torch.mean(loss) + L2 #torch.mean(logsum -linsum)  


def calc_loss(Yoh,fwd):
    """
    Calculates negative log likelihood loss
    Arguments:
    Yoh: One hot labels
    output: model.forward(x) 
    """
    loss = -torch.log(fwd + 1e-15)*Yoh
    loss = torch.sum(loss,dim = 1)
    return torch.mean(loss)



def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()
  probs = model(X)
  return probs.detach().numpy()




def train(model, X, Yoh_, param_niter, param_delta,param_lambda=1e-3,verbose = True):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """
  
  # inicijalizacija optimizatora
  
  optimizer = optim.SGD(model.parameters(),lr = param_delta)
  # petlja učenja
  # ispisujte gubitak tijekom učenja
  for i in range(param_niter):
    optimizer.zero_grad()

    loss = model.get_loss(X,Yoh_,param_lambda)
    
    if verbose:
      print(f"Epoch: {i+1} Loss: {loss.detach().numpy()}")
    
    loss.backward()

    optimizer.step()


def calc_metric(Y,Y_pred):
    """Calcualates performace metrics
        Arguments:
        Y: True labels
        Y_pred: Predicted labels

        Returns: Tuple
         A: class Accuracy vector
         P: class precision vector  
         R: class recall vector
         avgP: Average precision
    """
    CM = confusion_matrix(Y,Y_pred)
    cm_diag = np.diag(CM)
    N = len(Y)
    n_classes = CM.shape[0]
    P = np.zeros(n_classes)
    R = np.zeros(n_classes)
    F1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        TP = cm_diag[i]
        FP = np.sum(CM[i,:]) - TP
        FN = np.sum(CM[:,i])- TP
        P[i] = TP/(FP + TP)
        R[i] = TP/(FN + TP)
        F1[i] = 2*P[i]*R[i]/(P[i] + R[i])

    A = np.sum(cm_diag)/N
    avgP = np.sum(P)/n_classes
    for i in range(n_classes):
        print(f"class {i} \n F1 {F1[i]} \nPrecision {P[i]} \nRecall {R[i]}")
    print(f"Average Precision: {avgP}")
    print(f"Accuracy: {A}")


    return A,P,R,F1,avgP


if __name__ == '__main__':
    np.random.seed(100)
    X,Y_ = data.sample_gmm_2d(4, 3, 30)

    Yoh_ = nn.functional.one_hot(torch.tensor(Y_))  
    X = torch.tensor(X).float()
    

    model = PTDeep([2,3])
    
    train(model, X, Yoh_, 1000, param_delta =0.5,param_lambda= 0,verbose=True)
    
    probs = eval(model, X)
    rect=(np.min(X.detach().numpy(), axis=0), np.max(X.detach().numpy(), axis=0))
    data.graph_surface(lambda x: model.classify(torch.tensor(x).float()) ,rect, offset=0)
    
  # graph the data points
    data.graph_data(X.detach().numpy(), Y_,model.classify(X) , special=[])
    plt.show()
    model.count_params()

    sample1 = data.sample_gmm_2d(4,2,40)
    sample2 = data.sample_gmm_2d(6,2,10)
    samples = [sample1,sample2]

    config_1 = [2,2]
    config_2 = [2,10,2]
    config_3 = [2,10,10,2]
    configs = [config_1,config_2,config_3]
    activations = [torch.relu, torch.sigmoid]    
    
    frame, fig = plt.subplots(4,3,figsize = (20,15)) 
    for i,sample in enumerate(samples):
        
        X,Y_ = sample
        Yoh_ = data.class_to_onehot(Y_)
        X = torch.tensor(X,dtype = torch.float,requires_grad=False)
        Yoh_ = torch.tensor(Yoh_)

        for j,config in enumerate(configs):
            
            for k,activation in enumerate(activations):
                print("Model configuration: ",config)
                #Defining model
                model = PTDeep(config,activation = activation)
                lr = 1e-1
                if config == [2,10,10,2]:
                    verb = False
                    lr = 1e-3
                else:
                    verb = False
                
                
                #Training model
                train(model,X,Yoh_,int(1e4),param_delta = lr,param_lambda = 1e-4,verbose=verb)
                
                #Performing classification
                Y_pred = model.classify(X)

                
                #Calculating performance mertics
                A,P,R,F1,avgP = calc_metric(Y_,Y_pred)
                print("Class precision: ",P)
                print("Class recall: ",R)
                print("Class F1: ",F1)
                print(f"Accuracy:  {A}")
                print("Average precision: ",avgP)
                
                # graph the decision surface
                plt.subplot(4,3, 3*i +j + 6*k + 1)
                if k == 0:
                    act = "ReLU"
                else:
                    act = "Sigmoid"
                fig[i+2*k,j].set_title(act + " {}".format(config))
                rect=(np.min(X.detach().numpy(), axis=0), np.max(X.detach().numpy(), axis=0))
                data.graph_surface(lambda x: model.classify(torch.tensor(x).float()) ,rect, offset=0)
                data.graph_data(X.detach().numpy(), Y_,model.classify(X) , special=[])
    
    plt.show()
    plt.close()