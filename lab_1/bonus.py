from ast import With
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


import pt_deep
import data
import mnist_shootout as mnist



class WithBatchNorm(nn.Module):

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


        self.gamma = nn.Parameter(
            torch.ones(configuration_list[0]),
            requires_grad = True
        )
        self.beta = nn.Parameter(
            torch.zeros(configuration_list[0]),
            requires_grad = True
            )

        #E i var koji će biti korišteni pri zaključivanju
        #n je potreban za određivanje očekivane vrijednosti
        #E_smb i var_mb
        self.E = 0
        self.var = 0
        self.n = 0
        self.m = None #Batch size

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


    def batch_norm(self,x,eps = 1e-5,batch_size = None):#alpha = 0.9):


        if self.training:
            if batch_size is None and self.m is None:
                self.m = len(x)
            elif batch_size is not None and self.m is None:
                self.m = batch_size
            
            E_mb = torch.mean(x,dim  = 0)
            var_mb = torch.var(x,dim = 0)
            #Kumulativni prosijek očekivanja i varijance 
            # na minimatchevima - ovo funkcionira 
            # samo ako normalizaciju radim na podatcima
            self.E = (self.n*self.E + E_mb)/(self.n+1)
            self.var = (self.n*self.var + var_mb )/(self.n + 1)
            self.n += 1
            #U slučaju da provodim batch_norm na ulazu proizvoljnog sloja
            #koristio bih exponential moving avearage
            #self.E = alpha*self.E + (1-alpha)*E_mb
            #self.var = alpha*self.var + (1-alpha)*var_mb





            output = (x - E_mb)/torch.sqrt(var_mb + eps) 
            output *= self.gamma
            output += self.beta
            
            return output
        
        else:
            with torch.no_grad():
                V = self.m*self.var/(self.m-1)
                output = (x - self.E)/torch.sqrt(V + eps)
                output *= self.gamma
                output += self.beta
                return output


    def forward(self, x):
        fwd = self.batch_norm(x)
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



if __name__ == "__main__":
    np.random.seed(100)
    X,Y_ = data.sample_gmm_2d(4, 3, 30)

    Yoh_ = torch.tensor(data.class_to_onehot(Y_))  
    X = torch.tensor(X).float()
    

    model = WithBatchNorm([2,10,3],activation = torch.relu)
    
    pt_deep.train(model, X, Yoh_, 1000, param_delta = 0.5,param_lambda= 0,verbose=True)
    
    model.eval()
    probs = pt_deep.eval(model, X)
    rect=(np.min(X.detach().numpy(), axis=0), np.max(X.detach().numpy(), axis=0))
    data.graph_surface(lambda x: model.classify(torch.tensor(x).float()) ,rect, offset=0)
    
  # graph the data points
    data.graph_data(X.detach().numpy(), Y_,model.classify(X) , special=[])
    plt.show()
    model.count_params()

    (x_train,y_train),(x_test,y_test) = mnist.prep_data()
    mnist_model = WithBatchNorm([28*28,100,10],activation=torch.relu)
    batch_size = 32
    history,performance = mnist.train_mb(mnist_model,x_train,y_train,x_test,y_test,batch_size=batch_size,
    n_epochs=250,learning_rate = 1e-2)
    print(performance)
    with open("bonus_log.txt",'a') as f:
        f.write("Deep model with batch normalization performance on Test set\n")
        A,P,R,F1,avgP = performance
        for i in range(10):
            f.write(f"Class {i}\n")
            f.write(f"F1: {F1[i]}\n"+
            f"Precision: {P[i]}\n"+
            f"Recall: {R[i]}\n"
            )
        f.write(f"Accuracy: {A}\n")