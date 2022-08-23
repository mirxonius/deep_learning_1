import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix as confusion_matrix

import numpy as np  
import matplotlib.pyplot as plt
import data


class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super().__init__()


    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    self.W = nn.Parameter(torch.randn(D,C)) 
    self.b = nn.Parameter(torch.zeros(C))


  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    # ...
    phi = torch.mm(X,self.W)
    phi += self.b
    return torch.softmax(phi,dim = 1)


  def get_loss(self, X, Yoh_,param_lambda):
    # formulacija gubitka
    #   koristiti: torch.log, torch.mean, torch.sum
    L = -torch.sum(torch.log(self.forward(X))*Yoh_,axis = 1)
    L = torch.mean(L)
    L += param_lambda*torch.mm(self.W.T,self.W).sum()
    return L
    

  def classify(self,X):

    return self.forward(X).argmax(dim = 1).detach().numpy()


def train(model, X, Yoh_, param_niter, param_delta,param_lambda,verbose = True):
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
    
    loss = model.get_loss(X,Yoh_,param_lambda)
    
    if verbose:
      print(f"Epoch: {i+1} Loss: {loss.detach().numpy()}")
    
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

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




if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)
  D = 2
  C = 3
  # instanciraj podatke X i labele Yoh_  
  # get data
  X,Y_ = data.sample_gmm_2d(4, C, 30)

  Yoh_ = nn.functional.one_hot(torch.tensor(Y_))  
  X = torch.tensor(X).float()
  Yoh_ = torch.clone(Yoh_).detach()
  # definiraj model:

  models = [PTLogreg(D,C) for i in range(4)]
  lambdas = [0.8,1e-2,1e-4,0]
  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  frame,fig = plt.subplots(4,1, figsize = (5,20))

  probs = []
  for i, model in enumerate(models):
    train(model, X, Yoh_, 1000, 0.5,param_lambda= lambdas[i],verbose=False)
    probs.append(eval(model, X))

  # ispiši performansu (preciznost i odziv po razredima)

  # iscrtaj rezultate, decizijsku plohu
    Y = model(X)>0.5  
  
  # graph the decision surface
    plt.subplot(4,1,i+1)
    fig[i].set_title(r"$\lambda$ = {}".format(lambdas[i]))
    rect=(np.min(X.detach().numpy(), axis=0), np.max(X.detach().numpy(), axis=0))
    data.graph_surface(lambda x: model.classify(torch.tensor(x).float()) ,rect, offset=0)
    
  # graph the data points
    data.graph_data(X.detach().numpy(), Y_,model.classify(X) , special=[])
  
  for model in models:
    Y_pred = model.classify(X)
  
    CM = confusion_matrix(Y_,Y_pred)
    cm_diag = np.diag(CM)
    for i in range(C):
      TP = cm_diag[i]
      FP = np.sum(CM[i,:]) - cm_diag[i]
      FN = np.sum(CM[:,i])- cm_diag[i]
      
      print(f"Class label: {i+1}")
      print(f"Precision: {TP/(TP + FP)}")
      print(f"Recall: {TP / (TP + FN)}")

  plt.show()