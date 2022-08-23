import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
from torch.utils.data import DataLoader



class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=4, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE

        self.add_module(
            "bn",nn.BatchNorm2d(num_maps_in)
            )
        
        self.add_module(
            "relu",nn.ReLU()
            )
        
        self.add_module(
        "conv",nn.Conv2d(num_maps_in,num_maps_out,
        kernel_size = k,bias = bias)
        )


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32,margin = 1):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.margin = margin

        self.block1 = _BNReluConv(num_maps_in = input_channels,
        num_maps_out = emb_size)
        
        self.block2 = _BNReluConv(num_maps_in = emb_size,
        num_maps_out = emb_size)

        self.block3 = _BNReluConv(num_maps_in = emb_size,
        num_maps_out = emb_size)
        self.block4 = _BNReluConv(num_maps_in = emb_size,
        num_maps_out = emb_size)
        self.net = nn.Sequential(self.block1,
                                nn.MaxPool2d(kernel_size = 3, stride = 2),
                                self.block2,
                                nn.MaxPool2d(kernel_size = 3, stride = 2),
                                self.block3,
                                #self.block4,
                                #nn.AvgPool2d(kernel_size = 2, stride = 2)
                                    )


    def forward(self,img):
        return self.get_features(img)


    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.net(img)
        return torch.mean(x,dim = (2,3))


    def loss(self, anchor, positive, negative):
        """
        Triplet margin loss
        We use 2-norm to describe the distance between
        example embeddings
        """

        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        #Assuming a_x,p_x and n_x are all tenors
        #of type (N,D)
        d_ap = F.pairwise_distance(a_x,p_x)
        d_an = F.pairwise_distance(a_x,n_x)
        #d_ap i d_an alternativno: kao na predavanju
        #
        #d_ap = torch.cdist(a_x,p_x)
        #d_ap = d_ap.max(dim = 0)
        #d_an = torch.cdist(a_x,n_x)
        #d_an = d_an.min(dim = 0)

        loss = torch.maximum(d_ap - d_an + self.margin,torch.tensor(0))
        loss = torch.mean(loss)

        return loss




if __name__ == "__main__":

    ds = dataset.CIFARMetricDataset()

    model = SimpleMetricEmbedding(3)


    #ds = dataset.MNISTMetricDataset()

    #model = SimpleMetricEmbedding(1)
    a,p,n,id = ds[0]
    dl = DataLoader(ds,batch_size = 24,shuffle = True)
    batch = next(iter(dl))

    a,p,n,t = batch 
    print(a.shape)
    print(p.shape)
    print(n.shape)
    print("Batch feature vector: ",model.get_features(a).shape)
    print(
    model.loss(a,p,n)
    )