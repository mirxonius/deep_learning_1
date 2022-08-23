import copy
import os
from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from sklearn.svm import SVC

from pathlib import Path 
import pt_deep
import data

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent


def prep_data():
    """
    Loads MNIST train and test dataset
    Returns: (ds_train, ds_test)
    ds_train: training example, label tuple
    ds_test: test example, label tuple
    """
    mnist_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
   #train_mean = x_train.mean()
   # x_train, x_test = (x - train_mean for x in (x_train, x_test))

    ds_train = (x_train,y_train)
    ds_test = (x_test,y_test)
    return ds_train,ds_test



def task_1(X,Y,lr=1e-1,n_epochs = 10):
    """
    Arguments:
    X - training examlpes
    Y - labels
    """
    save_dir = os.path.join(SAVE_DIR,'task_1')
    #Regularizaijski parametri
    lambdas = [0.0,1e-4,1e-3,1e-2]

    Yoh = torch.tensor(data.class_to_onehot(Y))
    N = X.shape[0]
    W,H = X.shape[1],X.shape[2]
    D = W*H
    X = torch.clone(X).view(-1,D)

    config = [D,10]

    for lam in lambdas:

        model = pt_deep.PTDeep(config,activation=torch.relu)
        optimizer = optim.SGD(model.parameters(),lr = lr)

        for epoch in range(n_epochs):
            
            output = model.forward(X)            
            loss = pt_deep.calc_loss(Yoh,output) + model.L2_loss(lam)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        frame, fig  = plt.subplots(2,5,figsize = (20,15))
        for i in range(2):
            for j in range(5):
                fig[i,j].imshow(model.weights[0][:,5*i +j].view(W,H).detach().numpy())
                fig[i,j].set_title(f"{5*i + j }",fontsize = 15)

        if lam == 0:
            plt.savefig(os.path.join( save_dir,f"lambda_0.png" ))
            plt.close()           

        else:
            plt.savefig(os.path.join( save_dir,f"lambda_1e{int(np.log10(lam))}.png" ))
            plt.close()            


        


def task_2(X,Y,X_test,Y_test,learning_rate = 1e-1,n_epochs = 3):    
    save_dir = os.path.join(SAVE_DIR,'task_2')
    N = X.shape[0]
    W,H = X.shape[1],X.shape[2]
    D = W*H
    X = torch.clone(X).float().view(-1,D).detach()
    Yoh = torch.tensor(data.class_to_onehot(Y)).detach()

    X_test = torch.clone(X_test).float().view(-1,D).detach()
    Yoh_test = torch.tensor(data.class_to_onehot(Y_test)).detach()


    configs =[ [D,10], [D,100,10] ]
    model = [pt_deep.PTDeep(config,torch.relu) for config in configs]
    progress = []
    frame, fig = plt.subplots(figsize = (10,10))
    for i,config in enumerate(configs):
        if config == [D,100,10]:
            n_epochs = 3000
            learning_rate = 1e-2
        else:
            learning_rate = 1e-1

        print("Training model with configuration: ",config)
        optimizer = optim.SGD(model[i].parameters(),lr = learning_rate)

        train_history = []
        for epoch in range(n_epochs):
            
            output = model[i].forward(X)
            #output = output - torch.max(output,dim = 1).values.view(-1,1 ).detach()


            loss = pt_deep.calc_loss(Yoh,output)
            loss.backward()
            optimizer.step()

            train_history.append(
                loss.detach().numpy()
            )
            
            optimizer.zero_grad()

        
        Y_pred = model[i].classify(X_test)
        #pt_deep.calc_metric(Y_test,Y_pred)

        progress.append(train_history.copy())
        
        label = ''
        for j in range(len(config)):
            label += str(config[j])
            label += ' '

        


        fig.plot(range(1,n_epochs+1),progress[i],label = label)
        fig.set_xlabel("epoch",fontsize = 15)
    
    
    fig.set_title("Training loss",fontsize = 15) 
    fig.legend(fontsize = 15)
    plt.savefig(os.path.join(save_dir,'Loss function'))
    plt.close()

    #Tražim primjere s najvećom pogreškom
    best_idx = np.argmin([p[-1] for p in progress])
    best_model = model[best_idx]
    loss_per_eg = torch.tensor([best_model.get_loss(x.view(-1,D),y) for x,y in zip(X_test,Yoh_test)])
    worse_idx = torch.topk(loss_per_eg,3).indices
    frame, fig = plt.subplots(3,1,figsize = (8,24))
    for i,idx in enumerate(worse_idx):
        fig[i].imshow(X_test[idx].view(H,W).detach().numpy())
    plt.savefig(os.path.join(save_dir,"worse_examples"))
    plt.close()







def task_3(X,Y,X_test,Y_test,lambdas,lr = 1e-1,n_epochs = 3):
    #Effects of regularization on deep model performance

    save_dir = os.path.join(SAVE_DIR,'task_3')
    Yoh = torch.tensor(data.class_to_onehot(Y))
    N = X.shape[0]
    W,H = X.shape[1],X.shape[2]
    D = W*H
    X = torch.clone(X).detach().float().view(-1,D)

    X_test = torch.clone(X_test).detach().float().view(-1,D)
    Yoh_test = torch.tensor(data.class_to_onehot(Y_test))

    perm = torch.randperm(N)
    X = X[perm]
    Yoh = Yoh[perm]
    Y = Y[perm]

    configs = [[D,100,10]] # [D,100,100,10], [D,100,100,100,10] ] no gpu

    frame, fig = plt.subplots(figsize = (8,18))
    
    for i,config in enumerate(configs):

        for j,lam in enumerate(lambdas):
            model = pt_deep.PTDeep(config,activation=torch.relu)
            optimizer = optim.SGD(model.parameters(),lr = lr,weight_decay=lam)
            
            train_history = []
            for epoch in range(n_epochs):   
                optimizer.zero_grad()
        
                output = model.forward(X)            
                loss = pt_deep.calc_loss(Yoh,output)
                
                loss.backward()
                optimizer.step()

                train_history.append(
                loss.detach().numpy()
                )   


            with torch.no_grad():
                ypred = output.argmax(dim = 1).detach().numpy()
                Atr,P,R,F1,avgPtrain = pt_deep.calc_metric(Y,ypred)
                ypred = model.classify(X_test)
                Ate,P,R,F1,avgPtest = pt_deep.calc_metric(Y_test,ypred)

            fig.set_title("Training error",fontsize = 15)
            fig.plot(range(10 + 1,n_epochs+1),train_history[10:],
            label = r"$\lambda$ = {} train acc = {:0.3f} test acc {:0.3f}".format(lam,Atr,Ate))
            fig.set_xlabel("epoch",fontsize = 15)
    
            #fig[1].set_title("Test error",fontsize = 15)
            #fig[1].plot(range(1,n_epochs+1),test_history,
            #label = r"$\lambda$ = {} Accuracy = {:0.3f}".format(lam,np.mean(Ate)))
            #fig[1].set_xlabel("epoch",fontsize = 15)
    
    fig.legend(fontsize = 15)
    #fig[1].legend(fontsize = 15)
        
    plt.savefig(os.path.join(save_dir,'Loss function'))
    plt.close()



def task_4(model,X,Y,X_test,Y_test,val_size = 1/5,n_epochs = 3,
patience = 5,learning_rate = 0.1):

    #Shuffle data
    N,H,W = X.shape
    D = H*W
    perm = torch.randperm(N)
    X = X[perm]
    Y = Y[perm]
    #Data to tensors
    Yoh = torch.tensor(data.class_to_onehot(Y))
    X = torch.clone(X).detach().float().view(-1,D)
    
    X_test = torch.clone(X_test).detach().float().view(-1,D)
    #Train test split
    N_val = int(N*val_size)
    print(f"Train size: {N-N_val}\tValidation size {N_val}")

    X_val = X[:N_val]
    Yoh_val = Yoh[:N_val]
    Y_val = Y[:N_val]
    X = X[N_val:]
    Yoh = Yoh[N_val:]

    min_loss = None
    counter = 0
    valid_loss = None
    best_weights = model.weights
    best_biases = model.biases
    best_epoch = 0  

    optimizer = optim.SGD(model.parameters(),lr = learning_rate)
    for epoch in range(n_epochs):
        loss = model.get_loss(X,Yoh)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            valid_output = model.forward(X_val)
            
            valid_loss = pt_deep.calc_loss(Yoh_val,valid_output)
            if epoch % 100 == 0:
                print(f"EPOCH: {epoch +1}" )
                Y_pred = valid_output.argmax(dim = 1).detach().numpy()
                pt_deep.calc_metric(Y_val,Y_pred)


            if min_loss is None or min_loss > valid_loss:
                min_loss = valid_loss
                best_epoch = epoch + 1

                best_weights = model.weights
                best_biases = model.biases
                counter = 0

            else:
                counter += 1
                if counter >= patience:
                    model.weights = best_weights
                    model.biases = best_biases
                    break
    
    print(f"Best model is at epoch : {best_epoch}")
    print(f"Validation loss: {min_loss.item()}")
    print("Performance metrics on test set:")
    A,P,R,F1,avgP = pt_deep.calc_metric(Y_test,model.forward(X_test).argmax(dim = 1).detach().numpy())
    with open("task_4_log.txt",'w') as f:
        f.write(f"Best model is at epoch : {best_epoch}\n")
        f.write(f"Validation loss: {min_loss.item()}\n")
        for i in range(len(F1)):
            f.write(f"Class {i}\n")
            f.write(f"Accuracy {F1[i]}\n")
            f.write(f"Precision {P[i]}\n")
            f.write(f"Recall {R[i]}\n")

        f.write(f"Accuracy: {A}\n")
        f.write(f"Average precision {avgP}\n\n")



def train_mb(model,x_train,y_train,x_test,y_test,batch_size,n_epochs = 3,learning_rate = 1e-1):
    N,H,W = x_train.shape
    D = H*W
    C = y_train.max().add_(1).item()
    x_train = torch.clone(x_train).detach().view(-1,D)
    y_train = torch.tensor(data.class_to_onehot(y_train)).detach().view(-1,C)

    x_test = torch.clone(x_test).detach().view(-1,D)
    #y_test = torch.tensor(data.class_to_onehot(y_test)).detach().view(-1,C)
    

    history = []
    optimizer = optim.SGD(model.parameters(),lr = learning_rate)
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        x_train = x_train[perm]
        y_train = y_train[perm]
        x_train = torch.split(x_train,batch_size)
        y_train = torch.split(y_train,batch_size)

        epoch_history = []
        for x,y in zip(x_train,y_train):
            loss = model.get_loss(x,y)
            loss.backward()
            epoch_history.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

        history.append(np.mean(epoch_history))
        print(f"Epoch {epoch +1} Mean loss: {history[-1]}")

        x_train = torch.cat(x_train)
        y_train = torch.cat(y_train)


    print(f"Performance metrics on test set for batch size: {batch_size}")
    metrics = pt_deep.calc_metric(y_test,model.classify(x_test))

    return history,metrics




def task_5(config,x_train,y_train,x_test,y_test,n_epochs,batch_sizes=[32,64,128]):
    losses = []
    performance = []
    for batch_size in batch_sizes:
        print(f"Training model with bach sizes: {batch_size}")
        model = pt_deep.PTDeep(config,activation=torch.relu)
        history,metrics = train_mb(model,x_train,y_train,x_test,y_test,
        batch_size,n_epochs=n_epochs)

        performance.append(metrics)
        losses.append( 
             history
         )


    frame, fig = plt.subplots(figsize = (8,8) )
    for i,loss in enumerate(losses):
        fig.plot(range(1,n_epochs+1),np.log(np.array(loss)),label = f"batch size: {batch_sizes[i]}")

    fig.set_xlabel("Epoch",fontsize = 15)
    fig.set_title("log loss",fontsize = 15)
    fig.legend(fontsize = 15)
    plt.savefig("Task_8_subtask_5.png")
    plt.close()

    with open("task_5_log.txt",'w') as f:
        for i,batch_size in enumerate(batch_sizes):
            A,P,R,F1,avgP = performance[i]
            f.write(f"Batch size {batch_size}\n")
            for i in range(len(F1)):
                f.write(f"Class {i}\n")
                f.write(f"F1 {F1[i]}\n")
                f.write(f"Precision {P[i]}\n")
                f.write(f"Recall {R[i]}\n")
            f.write(f"Accuracy: {A}\n")
            f.write(f"Average precision {avgP}\n\n")




def task_6(config,x_train,y_train,x_test,y_test,n_epochs = 3):
    
    N,H,W = x_train.shape
    D = H*W
    C = y_train.max().add_(1).item()
    x_train = torch.clone(x_train).detach().view(-1,D)
    x_test = torch.clone(x_test).detach().view(-1,D)
    yoh_train = torch.tensor(data.class_to_onehot(y_train)).detach().view(-1,C)


    model = pt_deep.PTDeep(config,torch.relu)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    history = []
    for epoch in range(n_epochs):
        output = model.forward(x_train)
        loss = pt_deep.calc_loss(yoh_train,output)
        history.append(
            loss.detach().numpy()
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    
    
    y_pred = model.classify(x_test)
    A,P,R,F1,avgP = pt_deep.calc_metric(y_test,y_pred)

    frame,fig = plt.subplots(figsize =(8,8))
    fig.plot(range(1,n_epochs+1),history,label=f"Adam loss - test precision: {avgP:0.3f}")
    fig.legend(fontsize = 15)
    fig.set_xlabel("Epoch")
    plt.savefig("task_6_graph.png")
    plt.close()



def task_7(model,ds_train,ds_test,num_epochs = 3):
    x_train,y_train = ds_train
    x_test,y_test = ds_test
    N,H,W = x_train.shape
    D = H*W
    x_train = torch.clone(x_train).view(-1,D).detach()
    yoh_train = torch.tensor(data.class_to_onehot(y_train))

    x_test = torch.clone(x_test).view(-1,D).detach()
    yoh_test = torch.tensor(data.class_to_onehot(y_test))

    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 1-1e-4)

    history = list()
    for epoch in range(num_epochs):
        output = model.forward(x_train)
        loss = pt_deep.calc_loss(yoh_train,output)
        history.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    
    
    frame,fig = plt.subplots(figsize =(8,8))
    fig.plot(range(1,num_epochs+1),history,label=f"Loss with exponential scheduler")
    plt.savefig("task_7_graph.png")
    plt.close()

    with open("task_7_log.txt",'w') as f:
        A,P,R,F1,avgP = pt_deep.calc_metric(y_test,model.classify(x_test))

        f.write(f"Performance on test set\n")
        for i in range(len(F1)):
            f.write(f"Class {i}\n")
            f.write(f"Accuracy {F1[i]}\n")
            f.write(f"Precision {P[i]}\n")
            f.write(f"Recall {R[i]}\n")
        f.write(f"Accuracy: {A}\n")
        f.write(f"Average precision {avgP}\n\n")



def task_8(x_test,y_test):
    
    N,H,W = x_test.shape

    x_test = torch.clone(x_test).view(-1,H*W).detach()
    yoh_test = torch.tensor(data.class_to_onehot(y_test))
    model = pt_deep.PTDeep([H*W,100,10])
    loss = model.get_loss(x_test,yoh_test).detach().numpy()
    A,P,R,F1,avgP = pt_deep.calc_metric(y_test,model.classify(x_test))
    with open("random_model_log.txt",'w') as f:
        f.write(f"Model loss {loss}\n")
        f.write("Performance metrics")
        for i in range(len(F1)):
            f.write(f"Class {i}\n")
            f.write(f"F1 {F1[i]}\n")
            f.write(f"Precision {P[i]}\n")
            f.write(f"Recall {R[i]}\n")
        f.write(f"Accuracy: {A}\n")
        f.write(f"Average precision {avgP}\n\n")



def task_9(x_train,y_train,x_test,y_test):

    N,H,W = x_train.shape
    D = H*W
    C = y_train.max().add_(1).item()
    x_train = np.reshape(x_train.detach().numpy(),(N,D))
    y_train = y_train.detach().numpy()
    x_test = np.reshape(x_test.detach().numpy(),(x_test.shape[0],D))
    y_test = y_test.detach().numpy()

    linear = SVC(kernel='linear',decision_function_shape='ovo')
    rbf = SVC(decision_function_shape='ovo')
    
    linear.fit(x_train,y_train)
    rbf.fit(x_train,y_train)
    
    linear_pred = linear.predict(x_test)
    rbf_pred = rbf.predict(x_test)

    preds = [linear_pred,rbf_pred]
    labels = ["Linear SVM", "RBF SVM"]
    with open("task_9_log.txt",'w') as f:
        for j,pred in enumerate(preds):
            A,P,R,F1,avgP = pt_deep.calc_metric(y_test,pred)
            f.write("Performance metrics for " + labels[j])
            for i in range(len(F1)):
                f.write(f"Class {i}\n")
                f.write(f"F1 {F1[i]}\n")
                f.write(f"Precision {P[i]}\n")
                f.write(f"Recall {R[i]}\n")
                f.write(f"Accuracy: {A}\n")
                f.write(f"Average precision {avgP}\n\n")            



if __name__ == '__main__':
    np.random.seed(100)

    (x_train,y_train),(x_test,y_test) = prep_data()
    yoh_train = data.class_to_onehot(y_train)
    #yoh_test = data.class_to_onehot(y_test)
    
    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().add_(1).item()
    W = x_train.shape[1]
    H = x_train.shape[2]
    
    #x_train = torch.clone(x_train).view(-1,D).detach()
    #yoh_train = torch.tensor(yoh_train).detach()
    #x_test = torch.clone(x_test).view(-1,D).detach()

    #model = pt_deep.PTDeep([D,100,10],torch.relu)
    #pt_deep.train(model,x_train,yoh_train,param_niter = 1000,param_delta= 0.5,param_lambda=0)
    #ypred = model.classify(x_train)
    #pt_deep.calc_metric(y_train,ypred)
    #ypred = model.classify(x_test)
    #pt_deep.calc_metric(y_test,ypred)


    print("TASK 1")
    task_1(x_train,y_train,n_epochs = 1000)

    print("TASK 2")
    #task_2(x_train,y_train,x_test,y_test,n_epochs = 1000)
    
    print("TASK 3")
    #lambdas = [1e-3,1e-2,1e-1]
    #task_3(x_train,y_train,x_test,y_test,lambdas,n_epochs = 1000)

    print("TASK 4") 
    #model = pt_deep.PTDeep([D,100,10],activation = torch.relu)
    #task_4(model,x_train,y_train,x_test,y_test,n_epochs=10000)

    print("TASK 5")
    #task_5([D,100,10],x_train,y_train,x_test,y_test,n_epochs=250)

    print("TASK 6")
    #task_6([D,100,10],x_train,y_train,x_test,y_test,n_epochs=3000)

    print("TASK 7")
    #task_7(pt_deep.PTDeep([D,100,10],torch.relu),(x_train,y_train),(x_test,y_test),
    #num_epochs=3000)

    print("TASK 8")
    #task_8(x_test,y_test)

    print("TASK 9")
   #task_9(x_train,y_train,x_test,y_test)
