"""run.py: adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html"""
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



from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

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
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

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
        
    size = dist.get_world_size()
    bsz = 128 // size
#    partition_sizes = [1.0 / size for _ in range(size)]
    with open('partition_sizes', newline='') as csvfile1:
        partition_sizes = list(csv.reader(csvfile1))
    partition_sizes=[float(partition_sizes[0][i]) for i in range(size)]    
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True,drop_last=True)
    return train_set, bsz

def basic_average_gradients(model):
    """ Gradient averaging using Binomial Tree. """
#    print("Using DFL")
    global totaltime, starttime, endtime
    size = dist.get_world_size()
    rank = dist.get_rank()
    with open('layout-up', newline='') as csvfile1:
        btreedata1 = list(csv.reader(csvfile1))
    with open('layout-down', newline='') as csvfile2:
        btreedata2 = list(csv.reader(csvfile2))
    bytes_sent=0
    messages_sent=0
    for param in model.parameters():
#        if type(param) is torch.Tensor:
            model.mybuf=copy.deepcopy(param.grad.data)
#            model.testbuf=torch.tensor(np.zeros(1))
            #Tree Upward
#           for i in range(int(math.log2(size))):
#           for i in range(len(btreedata)):
            endtime=time.time()
            totaltime+=(endtime-starttime)
            for currentrow in btreedata1:
                         if int(currentrow[0]) == rank:
                           dist.send(tensor=param.grad.data,dst=int(currentrow[1]))
                           bytes_sent += param.grad.data.nelement() * param.grad.data.element_size()
                           messages_sent += 1
                         elif int(currentrow[1]) == rank:
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
                           param.grad.data+=model.mybuf

#Tree Downward

            for currentrow in btreedata2:
                        if int(currentrow[0]) == rank:
                           dist.send(tensor=param.grad.data,dst=int(currentrow[1]))
                           bytes_sent += param.grad.data.nelement() * param.grad.data.element_size()
                           messages_sent += 1

                        elif int(currentrow[1]) == rank:
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
                           param.grad.data=model.mybuf
#           dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            starttime=time.time()
            param.grad.data /= size
    return bytes_sent,messages_sent        

nextadjustment=None

def getnextadjustment(key):
    newstring=key
    while True:
        newstring=str(int(sha256(newstring.encode('utf-8')).hexdigest(),16))
        strlength=len(newstring)
        for i in range(strlength-4):
#           yield newstring[i:i+4]
#            yield '0.'+newstring[i:i+4]
            yield '0.'+newstring[i:i+4]
        newstring=newstring[strlength-4:strlength]
          

    
def set_leaf_pair_adder(rank, size, model):
    with open('layout-up', newline='') as csvfile1:
        btreedata1 = list(csv.reader(csvfile1))
    edge_dest=[currentrow[1] for currentrow in btreedata1]
    if str(rank) not in edge_dest:
        model.aux["isleaf"]=True
        if rank % 4 == 0:           
           model.aux["adder"]=True
           model.aux["partner"] = rank + 2
        else:
           model.aux["adder"]=False   
           model.aux["partner"] = rank - 2        
    else:
        model.aux["isleaf"]=False    
            

def my_average_gradients(model):
    """ Gradient averaging using Binomial Tree with SS """
#    print("Using DFL")
    global totaltime, starttime, endtime
    size = dist.get_world_size()
    rank = dist.get_rank()
    with open('layout-up', newline='') as csvfile1:
        btreedata1 = list(csv.reader(csvfile1))
    with open('layout-down', newline='') as csvfile2:
        btreedata2 = list(csv.reader(csvfile2))
    bytes_sent=0
    messages_sent=0
    
    global nextadjustment
        
    for param in model.parameters():
#        if type(param) is torch.Tensor:
            model.mybuf=copy.deepcopy(param.grad.data)
#            model.testbuf=torch.tensor(np.zeros(1))
#            additive = model.secret
            additive = 0.0
            if model.aux["isleaf"] == True:
                if model.aux["adder"] == True:
                    additive += float(next(nextadjustment))
                else:
                    additive -= float(next(nextadjustment))
            param.grad.data += additive
            endtime=time.time()
            totaltime+=(endtime-starttime)
#Tree Upward
#           for i in range(int(math.log2(size))):
#           for i in range(len(btreedata)):
            for currentrow in btreedata1:
                         if int(currentrow[0]) == rank:
                           dist.send(tensor=param.grad.data,dst=int(currentrow[1]))
                           bytes_sent += param.grad.data.nelement() * param.grad.data.element_size()
                           messages_sent += 1
                           
                         elif int(currentrow[1]) == rank:
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
                           param.grad.data+=model.mybuf

#Tree Downward

            for currentrow in btreedata2:
                        if int(currentrow[0]) == rank:
                           dist.send(tensor=param.grad.data,dst=int(currentrow[1]))
                           bytes_sent += param.grad.data.nelement() * param.grad.data.element_size()
                           messages_sent += 1
                        elif int(currentrow[1]) == rank:
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
                           param.grad.data=model.mybuf
#           dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            starttime=time.time()
            param.grad.data /= size
    return bytes_sent,messages_sent
def Add_SS(v, n, seed):
    vlist = []
    rnd=Random()
    rnd.seed(seed)
    r=[]
    for i in range(n):
        r += [rnd.random()]
    vstar = v/n
    for i in range(n-1):
        vlist += [vstar + r[i] - r[i+1]]
    vlist += [vstar + r[n-1] - r[0]]
    return vlist    

def their_average_gradients(model):
    """ Gradient averaging using modified LiPFed """
    global totaltime, starttime, endtime
    size = dist.get_world_size()
    rank = dist.get_rank()
    with open('layout', newline='') as csvfile1:
        btreedata1 = list(csv.reader(csvfile1))
    seedvalue=100
    bytes_sent=0
    messages_sent=0
    for param in model.parameters():
            first_receiving = True
            model.mybuf=copy.deepcopy(param.grad.data)
            model.splitbuf=copy.deepcopy(param.grad.data)

            #Split and Send

            edge_source=[currentrow[0] for  currentrow in btreedata1]

            #splits give the number of edges of a node - the number of splits of parameters, split[i]=edges connected to i
            splits=[edge_source.count(str(i)) for i in range(size)]
            
            #Send splits.. also receive :-)  Here was a bug when there were two for loops in place of the while..
            rowindex=0
            while rowindex < len(btreedata1):
                         if int(btreedata1[rowindex][0]) == rank:
                           number_of_splits=splits[rank]
                           splitparam=Add_SS(param.grad.data, number_of_splits, seedvalue)
                           endtime=time.time()
                           totaltime+=(endtime-starttime)
                           for j in range(number_of_splits):
                               targetnode=int(btreedata1[rowindex][1])
                               dist.send(tensor=splitparam[j],dst=targetnode)
                               bytes_sent += splitparam[j].nelement() * splitparam[j].element_size()
                               messages_sent += 1
                               rowindex += 1
                           starttime = time.time()    
                         elif int(btreedata1[rowindex][1]) == rank:
                           endtime=time.time()
                           totaltime+=(endtime-starttime)
                           dist.recv(tensor=model.splitbuf,src=int(btreedata1[rowindex][0]))
                           if first_receiving == True:
                                model.mybuf=copy.deepcopy(model.splitbuf)
                                first_receiving = False
                           else:     
                                model.mybuf+=model.splitbuf
                           rowindex += 1
                           starttime = time.time()
                         else:
                           endtime=time.time()
                           totaltime+=(endtime-starttime)
                           rowindex += 1
                           starttime = time.time()
#            dist.barrier()
            
            dist.all_reduce(model.mybuf, op=dist.reduce_op.SUM)
#           all reduce makes each node send model parameters at least log2(n) times
#            bytes_sent += math.log2(size) * model.mybuf.nelement() * splitparam[j].element_size()
#            messages_sent += math.log2(size)
            bytes_sent += size * model.mybuf.nelement() * model.mybuf.element_size()
            messages_sent += size

            param.grad.data = model.mybuf
            param.grad.data /= size
    return bytes_sent,messages_sent


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


#def run(rank, size):
#   """ Distributed function to be implemented later. """
#   print("Rank = ", rank)
def run(rank, size, epochs, K, averager, runid):
    """ Distributed Synchronous SGD Example """
    global totaltime, starttime, endtime
    torch.manual_seed(1234)
    seq_length=101
    BATCH_SIZE = 128//dist.get_world_size()
    path_to_file = 'shakespeare.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 128
    lr=0.001

    train_set, bsz = partition_dataset()
    model = RNN(input_size = vocab_size, output_size=seq_length-1, hidden_size=rnn_units, num_layers=3)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    LOG_FILE = "/logs/"+str(size)+"-"+averager+"-"+str(epochs)+"-"+str(runid)
    logging.basicConfig(filename=LOG_FILE, format='%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d_%H-%M-%S')
    starttime = time.time()
    
    global nextadjustment,device
    
    if averager == "DFLMSS":
        set_leaf_pair_adder(rank, size, model)
#        with open('secrets', newline='') as csvfile3:
#            thesecrets = list(csv.reader(csvfile3))
#            model.secret=float(thesecrets[rank][0])
        if model.aux["isleaf"] == True:
            with open('keys', newline='') as csvfile4:
                allkeys = list(csv.reader(csvfile4))
                model.aux["key"]=allkeys[rank//4][0]
            nextadjustment = getnextadjustment(model.aux["key"])
            
    runstarttime=time.time()
    total_bytes=0
    total_messgaes=0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        skip=0
        hidden_state = None
        for each_sequence in train_set:
            input_seq, target_seq =  split_input_target(each_sequence)
            input_batches=[[input_seq[i][j] for i in range(seq_length-1)] for j in range(bsz)]
            target_batches=[[target_seq[i][j] for i in range(seq_length-1)] for j in range(bsz)]
            input_seq = torch.tensor(input_batches).to(device)
            target_seq = torch.tensor(target_batches).to(device)
            output_seq, hidden_state = model(input_seq, hidden_state)
            loss = loss_fn(torch.squeeze(output_seq), torch.squeeze(target_seq))
            optimizer.zero_grad()
            epoch_loss += loss
            loss.backward()
            skip += 1
            if (skip % K) == 0:
               if averager == "DFLBASIC":
                  bytes_sent,messages_sent=basic_average_gradients(model)
                  total_bytes += bytes_sent
                  total_messgaes += messages_sent
               elif averager == "DFLMSS":
                  bytes_sent,messages_sent=my_average_gradients(model)                  
                  total_bytes += bytes_sent
                  total_messgaes += messages_sent
               elif averager == "DFLTSS":
                  bytes_sent,messages_sent=their_average_gradients(model)
                  total_bytes += bytes_sent
                  total_messgaes += messages_sent
            optimizer.step()
        print('Rank ',
            dist.get_rank(), ', epoch ', epoch, ': ',
            epoch_loss / num_batches)
        logging.info(f"Rank,{rank},epoch,{epoch},{epoch_loss/num_batches:.4f}")
    endtime = time.time()
    totaltime += (endtime - starttime)
    print(totaltime)
    runendtime=time.time()
    totalruntime=runendtime-runstarttime
    logging.info(f"Rank,{rank},TIME,{totaltime:.4f},OVERALL,{totalruntime:.4f},BYTES,{total_bytes},MESSAGES,{total_messgaes}")    



def init_processes(rank, size, epochs, K, averager, runid, fn, backend='gloo'):
   """ Initialize the distributed environment. """
   dist.init_process_group(backend, rank=rank, world_size=size)
   fn(rank, size, epochs, K, averager, runid)

if __name__ == "__main__":
#    rank=int(os.environ['LOCAL_RANK'])
    os.environ['GLOO_SOCKET_IFNAME']="eth0"
    os.environ['MASTER_ADDR'] = 'n0'
    os.environ['MASTER_PORT'] = '12321'    
#    rank=1
#    size=3
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--averager", type=str)
    parser.add_argument("--K", type=int)
    parser.add_argument("--runid", type=str)
    args = parser.parse_args()
    rank = int(args.rank)
    size = int(args.size)
    epochs = int(args.epochs)
    averager = args.averager
    K = int(args.K)
    runid = args.runid
    init_processes(rank, size, epochs, K, averager, runid, run)
