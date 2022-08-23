from collections import Counter,OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import csv



def get_freq(dataset):
    """
    dataset: pandas DataFrame object with two columns sentiment and label
    Returns:
    data_freq: frequency dictionary for words in dataset
    label_freq: frequency dictisonary for labels in dataset
    """
    data_freq = Counter()
    label_freq = Counter()
    cols = dataset.columns
    labels = dataset[cols[1]]
    sentiments = dataset[cols[0]]

    for label in labels:
        label_freq[label] += 1

    for setiniment in sentiments:

        words = setiniment.split(' ')
        for word in words:
            data_freq[word] += 1

    
    return data_freq,label_freq


def get_embedding_matrix(vocab,glove_path = "sst_glove_6b_300d.txt",D=300):

    if glove_path is not None:
        embedding_dict = dict()
        with open(glove_path,'r') as f:
            lines =f.read()
            lines = lines.split("\n")
            for line in lines:
                line = line.split(" ")
                token = line[0]
                vec = line[1:]
                vec = np.array([float(x) for x in vec])
                embedding_dict[token] = vec
        D = len(list(embedding_dict.values())[0])

    
    V = len(vocab.stoi)
    embedding_matrix = np.random.randn(V,D)
    embedding_matrix[0] = np.zeros( (1,D) ) #PAD znak postavljam na nulu
    if glove_path is not None:
        for i,word in enumerate(vocab.stoi):
            rep = embedding_dict.get(word,None)
            if rep is not None:
                embedding_matrix[i] = rep


    return embedding_matrix


def wrap_embedding_matrix(embedding_matrix,frozen = True):
    embedding_matrix = torch.tensor(embedding_matrix).detach()
    return torch.nn.Embedding.from_pretrained(embedding_matrix,freeze = frozen, padding_idx = 0)


    




class Vocab:

    def __init__(self,frequency_dict,max_size = None,min_freq = 1,special_symbols = None):
        
        if max_size == None:
            self.max_size = -1
        else:
            self.max_size = max_size

        self.min_freq = min_freq

        self.stoi = dict()
        self.itos = dict()
        
        n_special = 0
        if special_symbols is not None:
            for idx, token in enumerate(special_symbols):
                self.itos[idx] = token
                self.stoi[token] = idx
            n_special = len(special_symbols)

        

        frequency_dict = {k:v for k,v in frequency_dict.items() if v > self.min_freq}
        frequency_dict = sorted(frequency_dict,key = lambda i: frequency_dict[i],reverse=True)
        if self.max_size > 0 and self.max_size < len(frequency_dict) + n_special:
            to_remove = len(frequency_dict) - self.max_size + n_special
            del frequency_dict[-to_remove:]
                 

        for idx, token in enumerate(frequency_dict,n_special):
            self.stoi[token] = idx
            self.itos[idx] = token
        


    def encode(self,text,default = "<UNK>"):
        """
        text: list of tokens appearing in text
        Returns:
        encoded: list of indices of tokens appearing in text
        """
        if isinstance(text,list):
           return torch.tensor( [self.stoi.get(t,self.stoi[default]) for t in text] ,dtype = torch.int).detach()

        return torch.tensor(self.stoi.get(text, self.stoi[default]), dtype = torch.int ).detach()





class NLPDataset(Dataset):

    def __init__(self,data_file_path,special_symbols = ["<PAD>","<UNK>"],
        min_freq = 1,max_size = None):

        data = pd.read_csv(data_file_path,header = None)
        data.columns = ["sentiment","label"]
        sentiment_freq, label_freq = get_freq(data)


        self.instances = list()
        for i in range(data["label"].count()):
            
            sentiment = data["sentiment"].iloc[i].split(" ")
            label = data["label"].iloc[i]

            self.instances.append(
                (sentiment,label)
            )
    
        

        self.sentiment_vocab = Vocab(sentiment_freq,special_symbols = special_symbols,min_freq=min_freq,
        max_size=max_size)
        self.label_vocab = Vocab(label_freq)

    def __getitem__(self,index):
        instances = self.instances[index]
        
        encoded_sent = self.sentiment_vocab.encode(instances[0])
        encoded_labels = self.label_vocab.encode(instances[1],default = ' positive')
    
        return encoded_sent,encoded_labels


    def __len__(self):
        return len(self.instances)



def pad_collate_fn(batch,padd_index = 0):

    data, labels = zip(*batch)

    original_lengths = torch.tensor([len(eg) for eg in data])

    return (torch.nn.utils.rnn.pad_sequence(data, batch_first = True,padding_value = padd_index),
    torch.stack(labels,dim=0),
    original_lengths
    )




if __name__ == '__main__':

    print("Testiranje koda za zadatak 1")

    train = NLPDataset("sst_train_raw.csv")
    batch_size = 2
    shuffle = False
    dataloader = DataLoader(dataset=train, batch_size=batch_size, 
                              shuffle=shuffle,collate_fn=pad_collate_fn)

    texts, labels, lengths = next(iter(dataloader))
    print(f"Batch size : {batch_size}")
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")    
    
    embedder = get_embedding_matrix(train.sentiment_vocab)
    embedder = wrap_embedding_matrix(embedder)
    print(embedder(texts).shape )
    print(embedder(texts)[0])
