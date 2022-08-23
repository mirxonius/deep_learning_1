from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np

## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust


def linear_regression(data = None,num_epochs = 100, learning_rate = 0.1):

    if data == None:
        X = torch.tensor([1, 2])
        Y = torch.tensor([3, 5])
    else:
        X,Y = data

    a = torch.randn(1, requires_grad = True)
    b = torch.randn(1, requires_grad = True)

    optimizer = optim.SGD([a,b],lr = learning_rate)

    for i in range(num_epochs):
        y_pred = a*X + b

        loss = torch.mean((Y - y_pred).square())
        loss.backward()


        grada_analitic = -2*torch.mean( (Y - y_pred)* X)
        gradb_analitic = -2*torch.mean( Y - y_pred)
        print(f"Epoha: {i} Gubitak:{loss}")
        print(f"grad_a = {a.grad.detach().numpy()[0]}")
        print(f"grad_a analitic = {grada_analitic}")
        print(f"grad_b = {b.grad.detach().numpy()[0]}")
        print(f"grad_b analitic = {gradb_analitic}")
        
        optimizer.step()
        optimizer.zero_grad()



    return a,b


linear_regression() 