import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import namedtuple

from sklearn.metrics import confusion_matrix 
import numpy as np
import matplotlib.pyplot as plt

import task_1

seed = 7052020

class Baseline(nn.Module):

    def __init__(self,embedding_matrix,loss_fn = nn.BCEWithLogitsLoss()):
        """
        embedding_matrix: wrapped embedding_matrix
        """
        
        super().__init__()

        self.embedder = embedding_matrix

        self.fc = torch.nn.Sequential(
            nn.Linear(300,150), 
            nn.ReLU(), 
            nn.Linear(150,150), 
            nn.ReLU(),
            nn.Linear(150,1)
        )

        self.loss_fn = loss_fn
    

    def forward(self,x):
        """
        x: shape of x is (batch_size,T,D)
        """
        x = self.embedder(x)
        x_pooled = self.avg_pool(x)
        return self.fc(x_pooled).view(-1)


    def avg_pool(self,x):
        x_pooled = torch.sum(x,dim = 0 if len(x.shape)==2 else 1).detach().float()
        non_zero = torch.count_nonzero(x,dim = 1).detach().float()
        return x_pooled / non_zero

    def classify(self,x):
        logits = self.forward(x).detach().numpy()
        classes = np.zeros(y.shape)
        classes[logits > 0] += 1
        return classes

    
def train(model,data,optimizer,criterion,clip_value = None):
    
    loss_history = []

    model.train()
    for i, batch in enumerate(data):
        optimizer.zero_grad()
        x,y,lenghts = batch
        y = torch.clone(y).detach().float()
        logits = model(x)
        loss = criterion(logits,y)
    
        loss.backward()
        
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()

        loss_history.append(
            loss.detach().numpy()
        )
    
    return np.mean(np.array(loss_history))



def evaluate(model,data,criterion):
    val_history = []
    model.eval()
    CM = np.zeros([2,2]) #Radi se o binarnoj klasifikaciji
    with torch.no_grad():
        for i,batch in enumerate(data):
            x,y,lengths = batch
            y = torch.clone (y).detach().float()
            logits = model(x)

            val_history.append(
                criterion(logits,y).detach().numpy()
            )

            y_pred = np.zeros(y.shape)
            y_pred[logits.detach().numpy() > 0] += 1
            CM += confusion_matrix(y,y_pred)
            
    cm_diag = np.diagonal(CM)
    N = np.sum(CM)
    TP = cm_diag[0]
    TN = cm_diag[1]
    FP = CM[0,1]
    FN = CM[1,0]
    P = TP/(FP + TP)
    R = TP/(FN + TP)
    F1 = 2*P*R / ( P + R )
    Acc = (TP+TN)/N
    val_history = np.mean(np.array(val_history))
    
    return {
        "loss":val_history,
        "accuracy": Acc,
        "precision":P,
        "recall":R,
        "F1":F1
    }
    

def main(log_file,fig_name,batch_sizes,seed = seed,
n_epochs = 5,clip = None,lr = 1e-4,shuffle = True,):
    """
    log_file: File where to write model performance 
    """
    #Seed random variables for reproduciblilty
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_batch_size,validation_batch_size,test_bach_size = batch_sizes
    
    #Load datasets
    train_data = task_1.NLPDataset("sst_train_raw.csv")
    trainloader = DataLoader(dataset=train_data, batch_size=train_batch_size, 
                              shuffle=shuffle,collate_fn=task_1.pad_collate_fn)
    

    #Validation set and test set need to use the same
    #vocabulary as the test set!!!
    sentiment_vocab = train_data.sentiment_vocab
    label_vocab = train_data.label_vocab

    validation_data = task_1.NLPDataset("sst_valid_raw.csv")
    validation_data.sentiment_vocab = sentiment_vocab
    validation_data.label_vocab = label_vocab

    validloader = DataLoader(dataset = validation_data, 
    batch_size=validation_batch_size, shuffle=shuffle, collate_fn = task_1.pad_collate_fn)

    test_data = task_1.NLPDataset("sst_test_raw.csv")
    test_data.sentiment_vocab = sentiment_vocab
    test_data.label_vocab = label_vocab
    testloader = DataLoader(dataset = test_data,
    batch_size=test_bach_size, shuffle = shuffle, collate_fn=task_1.pad_collate_fn)

    #Define model
    embedder = task_1.get_embedding_matrix(train_data.sentiment_vocab)
    embedder = task_1.wrap_embedding_matrix(embedder)
    baseline = Baseline(embedder)
    optimizer = optim.Adam(baseline.parameters(),lr = lr)
    loss_fn = nn.BCEWithLogitsLoss()


    train_history = []
    valid_history = []
    with open(log_file,"a") as f:
        f.write("Basline model hyperparameters:\n")
        f.write(f"seed:{seed}, learning rate {lr}, total epochs:{n_epochs}\n")
        f.write(f"Train batch size:  {train_batch_size}, Validation batch size: {validation_batch_size}"\
            f"Test batch size: {test_bach_size}\n\n")
        for epoch in range(n_epochs):
            
            train_history.append(
            train(baseline,data = trainloader,
            optimizer = optimizer,criterion=loss_fn,clip_value=clip)
            )
            
            valid_performance = evaluate(baseline,validloader,loss_fn)
            valid_history.append(valid_performance["loss"])
            
            f.write(f"Performance on epoch: {epoch +1}\n")
            f.write(f"Train loss: { train_history[-1] }\n")
            f.write("Validation loss: {}\n".format(valid_performance["loss"]))
            f.write("Validation Accuacy: {}\n".format(valid_performance["accuracy"]))
            f.write("Validation Precision: {}\n".format(valid_performance["precision"]))        
            f.write("Validation Recall: {}\n".format(valid_performance["recall"]))
            f.write("Validation F1 measure: {}\n".format(valid_performance["F1"])) 
            f.write("\n\n")       

        test_performance = evaluate(baseline,testloader,loss_fn)
        f.write("Model performance on test set\n")
        f.write("Accuacy: {}\n".format(test_performance["accuracy"]))
        f.write("Precision: {}\n".format(test_performance["precision"]))        
        f.write("Recall: {}\n".format(test_performance["recall"]))
        f.write("F1 measure: {}\n".format(test_performance["F1"]))
    
    frame,fig = plt.subplots(figsize = (8,8))
    fig.plot(range(1,n_epochs+1),train_history,label = "Train loss")
    fig.plot(range(1,n_epochs+1),valid_history,label = "Validation loss")
    fig.set_xlabel("Epoch",fontsize = 15)
    fig.legend(fontsize = 15)
    plt.savefig(fig_name)
    plt.close()


if __name__ == '__main__':

    #Hyperparameters
    train_batch_size = 10
    validation_batch_size = 32
    test_bach_size = 32
    lr = 1e-4
    n_epochs = 5
    shuffle = True
    batch_sizes = [train_batch_size,validation_batch_size,test_bach_size]


    #Attempt 1
    seed1 = 7052020
    log_file1 = "task_2_attempt_1.txt"
    fig_name = "task_2_attempt_1.png"
    main(log_file=log_file1,fig_name=fig_name,batch_sizes=batch_sizes,
    n_epochs = n_epochs,seed=seed1)

    #Attempt 2
    seed2 = 100
    log_file1 = "task_2_attempt_2.txt"
    fig_name = "task_2_attempt_2.png"
    main(log_file=log_file1,fig_name=fig_name,batch_sizes=batch_sizes,
    n_epochs = n_epochs,seed=seed2)

    #Attempt 3
    seed3 = 123456
    log_file1 = "task_2_attempt_3.txt"
    fig_name = "task_2_attempt_3.png"
    main(log_file=log_file1,fig_name=fig_name,batch_sizes=batch_sizes,
    n_epochs = n_epochs,seed=seed3)

    #Attempt 4
    seed4 = 22111997
    log_file1 = "task_2_attempt_4.txt"
    fig_name = "task_2_attempt_4.png"
    main(log_file=log_file1,fig_name=fig_name,batch_sizes=batch_sizes,
    n_epochs = n_epochs,seed=seed4)

    #Attempt 5
    seed5 = 17032022
    log_file1 = "task_2_attempt_5.txt"
    fig_name = "task_2_attempt_5.png"
    main(log_file=log_file1,fig_name=fig_name,batch_sizes=batch_sizes,
    n_epochs = n_epochs,seed=seed5)