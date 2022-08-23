
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
        
        if self.has_attention:
            self.W1 = nn.Parameter(
                torch.randn(self.hidden_size,self.hidden_size), requires_grad = True
            )
            self.w2 = nn.Parameter(
                torch.randn(self.hidden_size,1), requires_grad = True
            )
        
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

        return description


    def attention(self,x):
        a = torch.mm(x,self.W1)
        a = torch.mm(x,self.w2)
        alpha = torch.softmax(a)
        return 0


    def forward(self,x):
        assert len(x.shape) == 2

        x = self.embedder(x).float()
        x = torch.transpose(x,dim0 = 0,dim1 = 1)
        if self.type == "LSTM":
            output, (h,c) = self.rnn(x)
            return self.decoder(h[-1].view(-1,self.hidden_size)).view(-1)
        
        output,h = self.rnn(x)
        out =  self.decoder(h[-1].view(-1,self.hidden_size)).view(-1)

        return out






def compare_rnn(hyperparam_list,log_file,seed = seed,batch_sizes = [10,32,32],
    shuffle = True,n_epochs = 5,clip_value = 0.25,lr = 1e-4):

    #Seed random variables for reproduciblilty
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    trainloader,validloader,testloader, datasets = get_data(batch_sizes=batch_sizes,
    shuffle=shuffle)

    train_data = datasets[0]
    #Define model
    embedder = task_1.get_embedding_matrix(train_data.sentiment_vocab)
    embedder = task_1.wrap_embedding_matrix(embedder)

    loss_fn = nn.BCEWithLogitsLoss()
    test_acc = []
    for i,hyperparams in enumerate(hyperparam_list):
        
        model = RecurrentModel(hyperparams=hyperparams,embedder=embedder)
        optimizer = optim.Adam(model.parameters(),lr = lr)
        with open(log_file,"a") as f:
            print("Training\n" + model.decribe_model())
            f.write("Model decription: " + model.decribe_model() + "\n")
            train_history = []
            valid_history = []
            for epoch in range(n_epochs):
                print(f"Epoch: {epoch + 1}")

                train_history.append(
                    task_2.train(model,data = trainloader,
                    optimizer = optimizer,criterion=loss_fn,clip_value=clip_value)
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
            test_acc.append(test_performance["accuracy"])
        
    best_i = np.argmax(test_acc)
    with open("best.txt",'a') as f:
        best = hyperparam_list[best_i]
        for param in best:
            f.write(param + "\t" + str(best[param] )+ "\n")

    return hyperparam_list[best_i]
            





def check_reproduciblity(hyperparams,log_file,fig_name,batch_sizes,seed = seed,
n_epochs = 5,clip = None,lr = 1e-4,shuffle = True):
    """
    log_file: File where to write model performance 
    """
    #Seed random variables for reproduciblilty
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_batch_size, validation_batch_size,test_batch_size = batch_sizes
    #Get data
    trainloader,validloader,testloader, datasets = get_data(batch_sizes=batch_sizes,
    shuffle=shuffle)

    train_data = datasets[0]

    #Define model
    embedder = task_1.get_embedding_matrix(train_data.sentiment_vocab)
    embedder = task_1.wrap_embedding_matrix(embedder)
    model = RecurrentModel(hyperparams,embedder)
    optimizer = optim.Adam(model.parameters(),lr = lr)
    loss_fn = nn.BCEWithLogitsLoss()


    train_history = []
    valid_history = []
    with open(log_file,"a") as f:
        f.write("Training\n" + model.decribe_model() + f"\n with seed {seed}")

        f.write(f"seed:{seed}, learning rate {lr}, total epochs:{n_epochs}\n")
        f.write(f"Train batch size:  {train_batch_size}, Validation batch size: {validation_batch_size}"\
            f"Test batch size: {test_batch_size}\n\n")
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





def optimize_hyperparams(hyperparam_list,model_hyperparams,log_file,
    seed = seed,batch_sizes = [10,32,32],shuffle = True,n_epochs = 5,lr = 1e-4):
    """
    hyperparams: lost training hyperparamter ditcionaries used for training
    These dictonaries include: 'min_freq', 'train_batchsize','optimizer',
    'frozen','clip value'
    model_hyperparams: hyperparam dict used to build model
    """
    #Seed random variables for reproduciblilty
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    for i, hyper in enumerate(hyperparam_list):
        batch_sizes[0] = hyper["train_batchsize"]
        trainloader,validloader,testloader, datasets = get_data(batch_sizes=batch_sizes,
        shuffle=shuffle,min_freq=hyper["min_freq"])
        train_data = datasets[0]
        embedder_rnn = task_1.get_embedding_matrix(train_data.sentiment_vocab,glove_path=None)
        embedder_rnn = task_1.wrap_embedding_matrix(embedder_rnn,frozen = hyper["frozen"])
        
        embedder_base = task_1.get_embedding_matrix(train_data.sentiment_vocab,glove_path=None)
        embedder_base = task_1.wrap_embedding_matrix(embedder_base,frozen = hyper["frozen"])
        
        rnn_model = RecurrentModel(model_hyperparams,embedder_rnn)
        baseline = task_2.Baseline(embedder_base)
        rnn_optimizer = hyper["optimizer"](rnn_model.parameters(),lr = lr)
        base_optimizer = hyper["optimizer"](baseline.parameters(),lr = lr)
        loss_fn = nn.BCEWithLogitsLoss()
        
        with open(log_file,'a') as f:
            f.write(
                "Training models with hyperparameters\n" +
                "min word frequency: " + str(hyper["min_freq"]) + "\n"
                + "training set batch size: " + str(hyper["train_batchsize"]) + "\n"
                +"Optimizer: " + str(hyper["optimizer"]) + "\n"
                + "Freeze embedding: " + str(hyper["frozen"]) + "\n"
                +"Gradient clipping value: " + str(hyper["clip_value"]) + "\n\n"
            )
            f.write("RNN model decription:\n" + rnn_model.decribe_model())

            for epoch in range(n_epochs):
                task_2.train(rnn_model,data = trainloader,optimizer = rnn_optimizer,
                criterion = loss_fn,clip_value = hyper["clip_value"])
                
                task_2.train(baseline,data = trainloader,optimizer = base_optimizer,
                criterion = loss_fn,clip_value = hyper["clip_value"])

                rnn_performance = task_2.evaluate(rnn_model,validloader,criterion=loss_fn)
                baseline_performance =task_2.evaluate(baseline,validloader,loss_fn)
                f.write(f"Epoch: {epoch + 1}\n")
                f.write("Baseline model validation accuracy: {}\n".format(baseline_performance["accuracy"]))
                f.write("Recurent model validation accuracy: {}\n".format(rnn_performance["accuracy"]))

            rnn_performance = task_2.evaluate(rnn_model,testloader,criterion=loss_fn)
            baseline_performance =task_2.evaluate(baseline,testloader,loss_fn)
            f.write(f"Test set performance\n")
            f.write("Baseline model test accuracy: {}\n".format(baseline_performance["accuracy"]))
            f.write("Recurent model test accuracy: {}\n".format(rnn_performance["accuracy"]))
            




    



    

if __name__ == "__main__":

    types = ["LSTM","GRU","RNN"]
    hidden_sizes = [10,150,300]
    num_layers = [1,2,5]
    dropouts = [0,0.1,0.5]
    fc_activ = [nn.ReLU, nn.Sigmoid,nn.SiLU]
    directions = [True,False]
    N_runs = 8
    hyperparam_list = []

    for i in range(N_runs):
        hyperparam_list.append(

            {"type": str(np.random.choice(types,replace = True)),
            "input_size":300, #Za sada ne mjenjamo reprezentaciju ulaza
            "hidden_size": int(np.random.choice(hidden_sizes,replace = True)),
            "num_layers":int(np.random.choice(num_layers,replace = True)),
            "bidirectional":bool(np.random.choice(directions,replace = True)),
            "fc_activation":np.random.choice(fc_activ,replace = True),
            "dropout":float(np.random.choice(dropouts,replace = True))
            }
        )

   # for h in hyperparam_list:
    #    print(h)
    #    for key in h:
  #          print(type(h[key]))
    
    log_file = "task_4_comparison.txt"
    best_hyperparams = compare_rnn(hyperparam_list=hyperparam_list,log_file=log_file)

    atempts = range(1,6)
    seeds = [100,420420,8769000,44332,1695212]
    file_name = "task_4_attempt_"
    fig_name = "task_4_attempt_"
    batch_sizes = [10,32,32]

    for atempt,seed in zip(atempts, seeds):
        check_reproduciblity(best_hyperparams,log_file = file_name + f"{atempt}.txt",
        fig_name=fig_name + f"{atempt}.png",
        batch_sizes=batch_sizes,
        seed = seed,
        clip=0.25
        )


    min_freqs = [2,10,100]
    train_batchsizes = [16,64,128] 
    optim_algorithms = [optim.Adam,optim.SGD]
    frozen = [True,False]    
    clip_values = [1e-3,0.5,10]
    n_runs = 8
    train_hyperparam_list = []
    for i in range(n_runs):
        train_hyperparam_list.append(
        {"min_freq": int(np.random.choice(min_freqs,replace = True)),
        "train_batchsize":int(np.random.choice(train_batchsizes),replace = True), 
        "optimizer": (np.random.choice(optim_algorithms,replace = True)),
        "frozen":bool(np.random.choice(frozen,replace = True)),
        "clip_value":float(np.random.choice(clip_values,replace = True)),
        }
    )
    
    log_file = "task_4_optimize_hyperparams.txt"
    optimize_hyperparams(train_hyperparam_list,best_hyperparams,
        log_file=log_file)