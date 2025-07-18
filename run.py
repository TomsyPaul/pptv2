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

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

from functools import reduce

from ftl.encryption import paillier, encryption


publickey=paillier.PaillierPublicKey(27236700922646976555595848507913589494886491119135730116014077185311243444450372255376489388880173022641848729747213088746475118480344996406749938547224285029951417411158327330610634671230458266993515963753271442282969744291116368707834837036890519842176076657317424175485854349519237230877898852294685281803161775833139254050216610420167131637216465657783550454961204111753470621658424459969937833601118914414496472033175054121693273513687334787107976849759736841476647931918984474457173711208172669939800415050356154977238127550304510079658979903408556459392897794799075517038480041829170731623511642064703877042081)


privatekey=paillier.PaillierPrivateKey(publickey,151731466574845229173039099025495922095008727037737139342593172671294559941585338353476978224816381692776568780654141379462769869960946855764863957672293447968060874024631012137102252157537274255768018381630660935684213060267736952413889789815316743297286676968944107787802792481731771876027538148701189447527,
179505949144779488794639217468096838709557457111534772428337237788171867936597162601187903575540819127792895820899862337184627927909110508331699702369829930039159510893967892328575714125966954175352601542034250713387089395862180071677366675266103958356045025432474981187150877764707656153946356889833385368503)

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


class Net(nn.Module):
    """ Network architecture. """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.mybuf=[]
        self.splitbuf=[]
#        self.secret=float(0)
        self.aux=dict(isleaf=False,partner=0,adder=False,key="1234567890")
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 // size
#    partition_sizes = [1.0 / size for _ in range(size)]
    with open('partition_sizes', newline='') as csvfile1:
        partition_sizes = list(csv.reader(csvfile1))
    partition_sizes=[float(partition_sizes[0][i]) for i in range(size)]    
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz

def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results


def aggregate_gradients(gradient_list, weight=0.5):
    # def multiply_by_weight(party, w):
    #     for i in range(len(party)):
    #         party[i] = w * party[i]
    #     return party

    # gradient_list = Parallel(n_jobs=2)(delayed(multiply_by_weight)(party, weight) for party in gradient_list)
    results = reduce(do_sum, gradient_list)
    return results


def aggregate_losses(loss_list):
    return np.sum(loss_list)




def basic_average_gradients(model):
    """ Gradient averaging using Binomial Tree. """
#    print("Using DFL")
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
            np_param_grad_data=param.grad.data.numpy()
            enc_grads_batch = [encryption.encrypt_matrix(publickey, x) for x in np_param_grad_data]
            model.mybuf=copy.deepcopy(enc_grads_batch)
#            model.testbuf=torch.tensor(np.zeros(1))
            #Tree Upward
#           for i in range(int(math.log2(size))):
#           for i in range(len(btreedata)):
            for currentrow in btreedata1:
                         if int(currentrow[0]) == rank:
                           dist.send(tensor=enc_grads_batch,dst=int(currentrow[1]))
                           bytes_sent += enc_grads_batch.nelement() * enc_grads_batch.element_size()
                           messages_sent += 1
                         elif int(currentrow[1]) == rank:
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
#                           param.grad.data+=model.mybuf
                           both_gradients=[]
                           both_gradients.append(enc_grads_batch)
                           both_gradients.append(model.mybuf)
                           enc_grads_batch = aggregate_gradients(both_gradients)
            if rank == size - 1:
                 dec_grads_batch=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in enc_grads_batch])
                 param.grad.data = torch.from_numpy(dec_grads_batch)    

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
                           for j in range(number_of_splits):
                               targetnode=int(btreedata1[rowindex][1])
                               dist.send(tensor=splitparam[j],dst=targetnode)
                               bytes_sent += splitparam[j].nelement() * splitparam[j].element_size()
                               messages_sent += 1
                               rowindex += 1
                         elif int(btreedata1[rowindex][1]) == rank:
                           dist.recv(tensor=model.splitbuf,src=int(btreedata1[rowindex][0]))
                           if first_receiving == True:
                                model.mybuf=copy.deepcopy(model.splitbuf)
                                first_receiving = False
                           else:     
                                model.mybuf+=model.splitbuf
                           rowindex += 1
                         else:
                           rowindex += 1       
#            dist.barrier()
            dist.all_reduce(model.mybuf, op=dist.reduce_op.SUM)
#           all reduce makes each node send model parameters at least log2(n) times
#            bytes_sent += math.log2(size) * model.mybuf.nelement() * splitparam[j].element_size()
#            messages_sent += math.log2(size)
            bytes_sent += size * model.mybuf.nelement() * splitparam[j].element_size()
            messages_sent += size

            param.grad.data = model.mybuf
            param.grad.data /= size
    return bytes_sent,messages_sent
#def run(rank, size):
#   """ Distributed function to be implemented later. """
#   print("Rank = ", rank)
def run(rank, size, epochs, K, averager, runid):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
#    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    LOG_FILE = "/logs/"+str(size)+"-"+averager+"-"+str(epochs)+"-"+str(runid)
    logging.basicConfig(filename=LOG_FILE, format='%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d_%H-%M-%S')
    starttime = time.time()
    
    global nextadjustment
    
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
            
    total_bytes=0
    total_messgaes=0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        skip=0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
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
    print(endtime - starttime)
    logging.info(f"Rank,{rank},TIME,{endtime-starttime:.4f},BYTES,{total_bytes},MESSAGES,{total_messgaes}")    



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
