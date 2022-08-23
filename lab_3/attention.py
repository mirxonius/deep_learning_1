from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import rnn

import task_1,task_2,task_3

import numpy as np
import matplotlib.pyplot as plt


#Default seed if none is given
seed = 7052020



def get_data(batch_sizes,shuffle = True,min_freq = 1,max_size = None):


    train_batch_size,validation_batch_size,test_bach_size = batch_sizes
    
    #Load datasets
    train_data = task_1.NLPDataset("sst_train_raw.csv",min_freq=min_freq,max_size=max_size)
    trainloader = DataLoader(dataset=train_data, batch_size=train_batch_size, 
                              shuffle=shuffle,collate_fn=task_1.pad_collate_fn)
    

    #Validation set and test set need to use the same
    #vocabulary as the test set!!!
    sentiment_vocab = train_data.sentiment_vocab
    label_vocab = train_data.label_vocab

    validation_data = task_1.NLPDataset("sst_valid_raw.csv",min_freq=min_freq,max_size=max_size)
    validation_data.sentiment_vocab = sentiment_vocab
    validation_data.label_vocab = label_vocab

    validloader = DataLoader(dataset = validation_data, 
    batch_size=validation_batch_size, shuffle=shuffle, collate_fn = task_1.pad_collate_fn)

    test_data = task_1.NLPDataset("sst_test_raw.csv",min_freq=min_freq,max_size=max_size)
    test_data.sentiment_vocab = sentiment_vocab
    test_data.label_vocab = label_vocab
    testloader = DataLoader(dataset = test_data,
    batch_size=test_bach_size, shuffle = shuffle, collate_fn=task_1.pad_collate_fn)


    return trainloader,validloader,testloader,(train_data,validation_data,test_data)




class RecurrentModel(nn.Module):

    def __init__(self,hyperparams,embedder,has_attention = False):
        """
        Makes a configurable RNN
        hyperparams: dictionary cointaing hyperparameters
        'type','input_size','hidden_size','num_layers',
        'bidirectional',
        'fc_activation', 'dropout',
        
        """
        super().__init__()

        self.embedder = embedder #Embedding matrix
        self.hyperparams = hyperparams
        self.type = hyperparams.get("type","RNN")
        self.fc_activation = hyperparams.get("fc_activation",nn.ReLU)
        self.has_attention = has_attention


        input_size = hyperparams["input_size"]
        self.hidden_size = hyperparams["hidden_size"]
        num_layers = hyperparams["num_layers"]
        bidirectional = hyperparams["bidirectional"] #Boolean
        
        


        if hyperparams["type"] == "LSTM":
            self.rnn = nn.LSTM(input_size = input_size,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            dropout = hyperparams.get("dropout",0) if num_layers != 1 else 0
            )

        elif hyperparams["type"] == "GRU":
            self.rnn = nn.GRU(input_size = input_size,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            bidirectional = hyperparams.get("bidirectional",False),
            dropout = hyperparams.get("dropout",0) if num_layers != 1 else 0
            )
        else:
            self.rnn = nn.RNN(input_size = input_size,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            bidirectional = hyperparams.get("bidirectional",False),
            dropout = hyperparams.get("dropout",0) if num_layers != 1 else 0
            )
            

        #Dekoder je isti u svakoj arhitekturi -  isti kao u zadatku 3.
        #uz varijabilnu aktivacijksu funkciju
        
        if bidirectional:
            self.hidden_size *= 2

        if self.has_attention:
            #self.W1 = nn.Parameter(
            #    torch.randn(self.hidden_size,self.hidden_size), requires_grad = True
            #)
            #self.w2 = nn.Parameter(
            #    torch.randn(self.hidden_size,1), requires_grad = True
            #)
            self.W1 = nn.Linear(self.hidden_size,self.hidden_size)
            self.w2 = nn.Linear(self.hidden_size,1)
        
            self.decoder = nn.Sequential(
                nn.Linear(2*self.hidden_size,150),
                self.fc_activation(),
                nn.Linear(150,1)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size,150),
                self.fc_activation(),
                nn.Linear(150,1)
            )
    


    def decribe_model(self):
        """
        Returns:
        description: string decribing model hyperparams
        """
        
        description = "Recurrent cell type: " + self.hyperparams["type"] + " With "+str(self.hyperparams["num_layers"]) + " layers" "\n"
        description += "Input size: " + str(self.hyperparams["input_size"]) + "\tHiden size: " + str(self.hyperparams["hidden_size"]) + "\n"
        description += "Bdirectional: " + str(self.hyperparams.get("bidirectional",False)) + "\tDropout rate: " + str(self.hyperparams.get("dropout",0)) + "\n"
        description += "Decoder activation: " + str(self.fc_activation) + "\n"
        if self.has_attention:
            description += "Model uses Bahdanau Attention"

        return description


    def attention(self,h):

        #a = torch.matmul(h,self.W1)
        #a = torch.matmul(h,self.w2)
        #alpha = torch.softmax(a,dim = 1)
        out = self.W1(h)
        out = torch.tanh(out)
        alpha = self.w2(out)
        alpha = torch.softmax(alpha,dim = 0)
        return torch.sum(alpha*h,dim = 0)


    def forward(self,x):
        assert len(x.shape) == 2

        x = self.embedder(x).float()
        x = torch.transpose(x,dim0 = 0,dim1 = 1)
        if self.type == "LSTM":
            output, (h,c) = self.rnn(x)
            if self.has_attention:
                attn = self.attention(output)
                hout = torch.cat(
                    (output[-1].view(-1,self.hidden_size),attn),dim = 1
                )
                return self.decoder(hout).view(-1)
            else:
                return self.decoder(h.view(-1,self.hidden_size)).view(-1)
        
        output,h = self.rnn(x)
        
        if self.has_attention:
            attn = self.attention(output)
            hout = torch.cat(
                (output[-1].view(-1,self.hidden_size),attn),dim = 1
                )
            out = self.decoder(hout).view(-1)
        
        else:    
            out =  self.decoder(output[-1].view(-1,self.hidden_size)).view(-1)
        return out



if __name__ == "__main__":

    torch.manual_seed(seed)
    np.random.seed(seed)
    shuffle = True
    batch_sizes = [10,32,32]
    trainloader,validloader,testloader, datasets = get_data(batch_sizes=batch_sizes,
    shuffle=shuffle)

    train_data = datasets[0]
    #Define model
    embedder = task_1.get_embedding_matrix(train_data.sentiment_vocab)
    embedder = task_1.wrap_embedding_matrix(embedder)
    hyperparams =  {"type": "LSTM",
            "input_size":300, #Za sada ne mjenjamo reprezentaciju ulaza
            "hidden_size": 150,
            "num_layers": 1 ,
            "bidirectional":True,
            "fc_activation":nn.ReLU,
            "dropout":0
            }


    model = RecurrentModel(hyperparams, embedder, has_attention = True)
    eg = next(iter(trainloader))
    print(eg[0].shape)
    print(model(eg[0]).shape)
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
        
    loss_fn = nn.BCEWithLogitsLoss()

    with open("attention","a") as f:
        print("Training\n" + model.decribe_model())
        f.write("Model decription: " + model.decribe_model() + "\n")
        train_history = []
        valid_history = []
        for epoch in range(5):
            print(f"Epoch: {epoch + 1}")

            train_history.append(
                task_2.train(model,data = trainloader,
                optimizer = optimizer,criterion=loss_fn,clip_value=0.25)
            )

            #Train performance is calculated to monitor overfitting
            #Large training, low validation and/or low train accuracy indicate overfittng
            train_performance = task_2.evaluate(model,trainloader,loss_fn)
            f.write(f"Epoch {epoch +1}: " + "Train accuracy: {}\n".format(train_performance["accuracy"]))

            valid_performance = task_2.evaluate(model,validloader,loss_fn)
            valid_history.append(valid_performance["loss"])
            f.write(f"Epoch {epoch + 1}: " + "Validation accuracy: {}\n".format(valid_performance["accuracy"])
                )
    
        test_performance = task_2.evaluate(model,testloader,loss_fn)
        f.write("Test accuracy: {}\n\n".format(test_performance["accuracy"]))
    