
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import rnn

import task_1,task_2

import numpy as np
import matplotlib.pyplot as plt


#Default seed if none is given
seed = 7052020


class Model(nn.Module):


    def __init__(self,embedding_matrix,loss_fn = nn.BCEWithLogitsLoss() ):
        super().__init__()

        self.embedder = embedding_matrix
        
        self.rnn1 = nn.LSTM(input_size = 300,
        hidden_size = 150,num_layers = 2)
        
        #self.rnn2 = nn.LSTM(input_size = 150,
        #hidden_size = 150,num_layers = 2)

        self.decoder = nn.Sequential(
            nn.Linear(150,150),
            nn.ReLU(),
            nn.Linear(150,1)
        )


    def forward(self,x,lengths = None):

        x = self.embedder(x).float()
        x = torch.transpose(x,dim0 = 0,dim1= 1)
        if lengths is not None:

            x = rnn.pack_padded_sequence(x,lengths,enforce_sorted = False)
            packed_out,(h,c) = self.rnn1(x)
            
            return self.decoder(h[-1].view(-1,150)).view(-1)
        

        output, (h,c) = self.rnn1(x)
        #U dekoder šaljem skriveno stanje zadnjeg sloja LSTM-a
        return self.decoder(h[-1].view(-1,150)).view(-1)


def main(log_file,fig_name,batch_sizes,seed = seed,
n_epochs = 5,clip = None,lr = 1e-4,shuffle = True):
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
    model = Model(embedder)
    optimizer = optim.Adam(model.parameters(),lr = lr)
    loss_fn = nn.BCEWithLogitsLoss()


    train_history = []
    valid_history = []
    with open(log_file,"a") as f:
        f.write("LSTM model hyperparameters:\n" +
        "Num hidden rnn layers: 2\n")

        f.write(f"seed:{seed}, learning rate {lr}, total epochs:{n_epochs}\n")
        f.write(f"Train batch size:  {train_batch_size}, Validation batch size: {validation_batch_size}"\
            f"Test batch size: {test_bach_size}\n\n")
        for epoch in range(n_epochs):
            
            train_history.append(
            task_2.train(model,data = trainloader,
            optimizer = optimizer,criterion=loss_fn,clip_value=clip)
            )
            
            valid_performance = task_2.evaluate(model,validloader,loss_fn)
            valid_history.append(valid_performance["loss"])
            
            f.write(f"Performance on epoch: {epoch +1}\n")
            f.write(f"Train loss: { train_history[-1] }\n")
            f.write("Validation loss: {}\n".format(valid_performance["loss"]))
            f.write("Validation Accuacy: {}\n".format(valid_performance["accuracy"]))
            f.write("Validation Precision: {}\n".format(valid_performance["precision"]))        
            f.write("Validation Recall: {}\n".format(valid_performance["recall"]))
            f.write("Validation F1 measure: {}\n".format(valid_performance["F1"])) 
            f.write("\n\n")       

        test_performance = task_2.evaluate(model,testloader,loss_fn)
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



if __name__ == "__main__":

    atempts = range(1,6)
    seeds = [seed,420420,8769000,44332,1695212]
    file_name = "task_3_attempt_"
    fig_name = "task_3_attempt_"
    batch_sizes = [10,32,32]

    for atempt,seed in zip(atempts, seeds):
        main(log_file = file_name + f"{atempt}.txt",
        fig_name=fig_name + f"{atempt}.png",
        batch_sizes=batch_sizes,
        seed = seed,
        clip=0.25
        )