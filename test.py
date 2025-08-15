#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import csv
import copy
import logging
import time
from hashlib import sha256

totaltime=0
starttime=0
endtime=0

device = "cpu"

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        #rng = Random()
        #rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        #rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.mybuf=[]
        self.splitbuf=[]
        self.aux=dict(isleaf=False,partner=0,adder=False,key="1234567890")
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def partition_dataset():
    """ Partitioning Shakespeare """
    data_path = 'shakespeare.txt'
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    seq_length = 101
    data_set=[]
    
    for i in range((len(data)//seq_length)):
        source=data[i*seq_length:(i+1)*seq_length]
        data_set+=[source]
    # data tensor on device
    #data = torch.tensor(data_set).to(device)
    #data = torch.unsqueeze(data_set, dim=1)
    size = 4
    rank = 1
#    bsz = 64 // size
    bsz=128
    #breakpoint()
    with open('partition_sizes', newline='') as csvfile1:
        partition_sizes = list(csv.reader(csvfile1))
    partition_sizes=[float(partition_sizes[0][i]) for i in range(size)]    
    partition = DataPartitioner(data_set, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=False,drop_last=True)
    return train_set, bsz


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#def run(rank, size):
#   """ Distributed function to be implemented later. """
#   print("Rank = ", rank)
def run(rank, size, epochs, K, averager, runid):
    """ Distributed Synchronous SGD Example """

    seq_length=101
    path_to_file = 'shakespeare.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
   
    # Batch size
    BATCH_SIZE = 128
    #steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    #rnn_units = 128
    rnn_units = 128
    lr=0.001

    train_set, bsz = partition_dataset()

    #data = torch.tensor(data).to(device)
    #data = torch.unsqueeze(data, dim=1)

    model = RNN(input_size = vocab_size, output_size=seq_length-1, hidden_size=rnn_units, num_layers=3)
                
    global totaltime, starttime, endtime
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    
#    LOG_FILE = "/logs/"+str(size)+"-"+averager+"-"+str(epochs)+"-"+str(runid)
#    logging.basicConfig(filename=LOG_FILE, format='%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d_%H-%M-%S')
    starttime = time.time()
    
    global nextadjustment
    
#    if averager == "DFLMSS":
#        set_leaf_pair_adder(rank, size, model)

#        with open('secrets', newline='') as csvfile3:
#            thesecrets = list(csv.reader(csvfile3))
#            model.secret=float(thesecrets[rank][0])

#        if model.aux["isleaf"] == True:
#            with open('keys', newline='') as csvfile4:
#                allkeys = list(csv.reader(csvfile4))
#                model.aux["key"]=allkeys[rank//4][0]
#            nextadjustment = getnextadjustment(model.aux["key"])
            
    runstarttime=time.time()
    total_bytes=0
    total_messgaes=0
    #seq_length=101
    for epoch in range(epochs):
        epoch_loss = 0.0
        skip=0
        hidden_state = None
        for each_sequence in train_set:
            #data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            #breakpoint()
            input_seq, target_seq =  split_input_target(each_sequence)
            input_batches=[[input_seq[i][j] for i in range(seq_length-1)] for j in range(bsz)]
            target_batches=[[target_seq[i][j] for i in range(seq_length-1)] for j in range(bsz)]
            input_seq = torch.tensor(input_batches).to(device)
            #input_seq = torch.unsqueeze(input_seq, dim=1)
            target_seq = torch.tensor(target_batches).to(device)
            #target_seq = torch.unsqueeze(target_seq, dim=1)

            output_seq, hidden_state = model(input_seq, hidden_state)
            
            loss = loss_fn(torch.squeeze(output_seq), torch.squeeze(target_seq))
            epoch_loss += loss.item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            
            #loss = F.nll_loss(output, target)
            
            loss.backward()
            skip += 1
            optimizer.step()
        print('Rank ',
            1, ', epoch ', epoch, ': ',
            epoch_loss / num_batches)
        #logging.info(f"Rank,{rank},epoch,{epoch},{epoch_loss/num_batches:.4f}")
    print('Num Batches',num_batches)    
    endtime = time.time()
    totaltime += (endtime - starttime)
    print(totaltime)
    runendtime=time.time()
    totalruntime=runendtime-runstarttime
    logging.info(f"Rank,{rank},TIME,{totaltime:.4f},OVERALL,{totalruntime:.4f},BYTES,{total_bytes},MESSAGES,{total_messgaes}")    




if __name__ == "__main__":
   run(1,4,10,2,"DFLMSS",1234)
