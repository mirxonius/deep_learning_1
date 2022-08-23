import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import data


class fcann2:

    def __init__(self,input_features,hidden_features,n_classes):
         
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.n_classes = n_classes

        self.W1 = self.initialize_weights(input_features,hidden_features)
        self.b1 = np.zeros([1,self.hidden_features])
        self.W2 = self.initialize_weights(self.hidden_features,self.n_classes)
        self.b2 = np.zeros([1,self.n_classes])
    
    def initialize_weights(self,rows,cols = None):
        """Generates a out_dim x in_dim 
           random matrix
        """
        
        return np.random.randn(rows,cols)


    
    
    def ReLU(self,x):

        return np.maximum(np.zeros(x.shape),x)
    

    def forward(self,x):
        """Calculates class probabilites 
           for input x  
           Arguemtns:
           x
           Returns:
           P: class probability vector

        """
        s1 = np.matmul(x,self.W1) +  self.b1
        h1 = self.ReLU(s1)
        s2 = np.matmul(h1,self.W2) + self.b2
        exp_s2 = np.exp(s2)
        exp_s2_sum = np.sum(exp_s2,axis = 1)
        P = exp_s2 / exp_s2_sum[:,np.newaxis]
        return P



    def classify(self,x):
        """Arguments:
            x: design matrix o
            Returns:
                class for each data point in x
        
        """
        return np.argmax(self.forward(x),axis = 1)

    def calc_loss(self,X,Y,param_lambda = 1e-3):
        """Arguments:
        X: Design matrix N x D
        Y: Data labels NxC
        param_lambda: regularization parameter
        
        Returns:
        L: model loss on X given Y
        """
        N = X.shape[0]
        logP = np.log(self.forward(X))
        log_likelihood = Y*logP # Množenje član po član
        L = -np.sum(log_likelihood)/N
        L += param_lambda * np.sum( np.matmul(self.W1.T,self.W1) )
        L += param_lambda * np.sum(np.matmul(self.W2.T,self.W2 ) )
        return L



    def calc_relu_grad(self,s):
        one = np.ones(s.shape)
        zero = np.zeros(s.shape)
        return np.where(s>0,one,zero)

    def decision_boundary(self,X):
        return self.forward(X)[:,0]


    def train(self,X,Y,param_niter = 1e3,param_delta = 0.05,param_lambda = 1e-3):
        """Trains the feedforward model
           
           Arguments:
           X: data points
           Y: data class labels
           param_niter: number of training iterations
           param_delta: learning rate
           param_lambda: regularization factor

           Returns:
            None
           
        """
        for it in range(int(param_niter)):
            
            s1 = np.matmul(X,self.W1) +  self.b1
            h1 = self.ReLU(s1)
            s2 = np.matmul(h1,self.W2) + self.b2
            exp_s2 = np.exp(s2)
            exp_s2_sum = np.sum(exp_s2,axis = 1)
            Gs2 = exp_s2 / exp_s2_sum[:,np.newaxis]
            
            Gs2[np.arange(len(X)),Y.argmax(axis=-1)] -=1

            Gs2 /= len(X)
            gradW2 = np.matmul(h1.T,Gs2) + param_lambda*self.W2
            #gradW2 += param_lambda*self.W2
            gradb2 = np.sum(Gs2,axis = 0 )
            
            Gs1 = np.matmul(Gs2,self.W2.T)*self.calc_relu_grad(s1)

            gradW1 = np.matmul(X.T,Gs1) + param_lambda*self.W1
 
            #gradW1 += param_lambda*self.W1
            gradb1 = np.sum(Gs1,axis = 0)

            self.W1 -= param_delta*gradW1

            self.b1 -= param_delta*gradb1

            self.W2 -= param_delta*gradW2

            self.b2 -= param_delta*gradb2
            #print(gradW2)
            print("iteration: ",it,"loss: ",self.calc_loss(X,Y))
            




if __name__ == "__main__":
    np.random.seed(100  )
    X,Y_ = data.sample_gmm_2d(6,2,10)
    Yoh = data.class_to_onehot(Y_)     
    net = fcann2(2,5,2)
    #print(net.forward(X))
    
    net.train(X,Yoh,param_niter=1e5 )
    
    
    Y = net.forward(X)[:,0]>0.5
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface( net.decision_boundary , rect, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()


