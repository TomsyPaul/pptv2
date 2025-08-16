"""run.py: adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html"""
#!/usr/bin/env python
import pickle
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

from joblib import Parallel, delayed
import multiprocessing

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

from functools import reduce

from ftl.encryption import paillier, encryption

N_JOBS = multiprocessing.cpu_count()

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
    partition = DataPartitioner(data_set, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True,drop_last=True)
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


def batch_enc_per_layer(publickey, party, r_maxs, bit_width=16, batch_size=100):
    result = []
    og_shapes = []

    for layer, r_max in zip(party, r_maxs):
        enc, shape_ = encryption.encrypt_matrix_batch(publickey, layer, batch_size=batch_size, bit_width=bit_width,
                                                      r_max=r_max)
        result.append(enc)
        og_shapes.append(shape_)
    return result, og_shapes


def batch_dec_per_layer(privatekey, party, og_shapes, r_maxs, bit_width=16, batch_size=100):
    result = []
    for layer, r_max, og_shape in zip(party, r_maxs, og_shapes):
        result.append(
            encryption.decrypt_matrix_batch(privatekey, layer, og_shape, batch_size=batch_size, bit_width=bit_width,
                                            r_max=r_max).astype(np.float32))
    return result


def quantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        x, _ = encryption.quantize_matrix(component, bit_width=bit_width, r_max=r_max)
        result.append(x)
    return np.array(result)


def quantize_per_layer(party, r_maxs, bit_width=16):
    # result = []
    # for component, r_max in zip(party, r_maxs):
    #     x, _ = encryption.quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
    #     result.append(x)
    result = Parallel(n_jobs=N_JOBS)(
        delayed(encryption.quantize_matrix_stochastic)(component, bit_width=bit_width, r_max=r_max) for component, r_max
        in zip(party, r_maxs))
    result = np.array(result)[:, 0]
    return result


def unquantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return np.array(result)


def unquantize_per_layer(party, r_maxs, bit_width=16):
    # result = []
    # for component, r_max in zip(party, r_maxs):
    #     result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    result = Parallel(n_jobs=N_JOBS)(
        delayed(encryption.unquantize_matrix)(component, bit_width=bit_width, r_max=r_max) for component, r_max
        in zip(party, r_maxs)
    )
    return np.array(result)


def basic_average_gradients_cq_ben(model,root):
    """ Gradient averaging using Binomial Tree., Batch Crypt with aciq-quan and enc """
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
            if param.dim() == 1:
              continue
            num_clients=size
            grads_batch_clients=[param.grad.data.numpy()]
            q_width=16

            sizes = [item.size * num_clients for item in grads_batch_clients[0]]
            max_values = []
            min_values = []
            for layer_idx in range(len(grads_batch_clients[0])):
                max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
                min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
            max_values=torch.tensor(max_values)
            min_values=torch.tensor(min_values)
            dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_values, op=dist.ReduceOp.MIN)
            grads_max_min = np.concatenate([max_values.numpy(),min_values.numpy()],axis=1)
            clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)
            #print("clipping_thresholds", clipping_thresholds)


            r_maxs = [x * num_clients for x in clipping_thresholds]

            grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                    for item in grads_batch_clients]

            grads_batch_this = [quantize_per_layer(item, r_maxs, bit_width=q_width)
                                    for item in grads_batch_clients][0]
                                    
            enc_grads_batch_clients = []
            og_shape_batch_clients = []
            batch_size=100
            
            for item in grads_batch_clients:
                enc_grads_temp, og_shape_temp = batch_enc_per_layer(publickey=publickey, party=item,
                                                                    r_maxs=r_maxs,
                                                                    bit_width=q_width,
                                                                    batch_size=batch_size)
                enc_grads_batch_clients.append(enc_grads_temp)
                og_shape_batch_clients.append(og_shape_temp)
           
            grads_batch_this = enc_grads_batch_clients[0]
            
            grads_batch_this_serialised=pickle.dumps(grads_batch_this)
            tensor_to_send = torch.ByteTensor(list(grads_batch_this_serialised))
            #model.mybuf=copy.deepcopy(tensor_to_send)
            
            #breakpoint()    
            
#            model.testbuf=torch.tensor(np.zeros(1))
            #Tree Upward
#           for i in range(int(math.log2(size))):
#           for i in range(len(btreedata)):
            endtime=time.time()
            totaltime+=(endtime-starttime)
            for currentrow in btreedata1:
                         #logging.info(f"Rank,{rank},currentrow,{currentrow[0],currentrow[1]}")
                         if int(currentrow[0]) == rank:
                           dist.send(tensor=torch.tensor(len(tensor_to_send),dtype=torch.int64),dst=int(currentrow[1]))
                           dist.send(tensor=tensor_to_send,dst=int(currentrow[1]))
                           bytes_sent += len(tensor_to_send)
                           messages_sent += 1
                         elif int(currentrow[1]) == rank:
                           temp_tensor=torch.tensor(0,dtype=torch.int64)
                           dist.recv(tensor=temp_tensor,src=int(currentrow[0]))
                           model.mybuf=torch.ByteTensor(temp_tensor.item())
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
#                          param.grad.data+=model.mybuf
                           received_bytes = bytes(model.mybuf.tolist())
                           grads_batch_that=pickle.loads(received_bytes)
                           #cipherz = pickle.loads(received_bytes)
                           both_gradients=[]
                           both_gradients.append(grads_batch_this)
                           both_gradients.append(grads_batch_that)
                           starttime=time.time()
                           grads_batch_this = aggregate_gradients(both_gradients)
                           endtime=time.time()
                           totaltime+=(endtime-starttime)
                           grads_batch_this_serialised=pickle.dumps(grads_batch_this)
                           tensor_to_send = torch.ByteTensor(list(grads_batch_this_serialised))
                         
            if rank == root:
                 starttime=time.time()
                 grads_batch_temp = batch_dec_per_layer(privatekey=privatekey, party=grads_batch_this, og_shapes=og_shape_batch_clients[0],
                                            r_maxs=r_maxs, bit_width=q_width, batch_size=batch_size)
                 grads_batch_final=unquantize_per_layer(grads_batch_temp, r_maxs, bit_width=q_width)
                 endtime=time.time()
                 totaltime+=(endtime-starttime)
                 param.grad.data = torch.from_numpy(grads_batch_final)    

#Tree Downward
            model.mybuf=copy.deepcopy(param.grad.data)
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

def basic_average_gradients_cq(model,root):
    """ Gradient averaging using Binomial Tree., Batch Crypt with aciq-quan """
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
            if param.dim() == 1:
              continue
            num_clients=size
            grads_batch_clients=[param.grad.data.numpy()]
            q_width=16

            sizes = [item.size * num_clients for item in grads_batch_clients[0]]
            max_values = []
            min_values = []
            for layer_idx in range(len(grads_batch_clients[0])):
                max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
                min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
            max_values=torch.tensor(max_values)
            min_values=torch.tensor(min_values)
            dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_values, op=dist.ReduceOp.MIN)
            grads_max_min = np.concatenate([max_values.numpy(),min_values.numpy()],axis=1)
            clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)
            #print("clipping_thresholds", clipping_thresholds)


            r_maxs = [x * num_clients for x in clipping_thresholds]

            grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                    for item in grads_batch_clients]

            grads_batch_this = [quantize_per_layer(item, r_maxs, bit_width=q_width)
                                    for item in grads_batch_clients][0]
            
            grads_batch_this_serialised=pickle.dumps(grads_batch_this)
            tensor_to_send = torch.ByteTensor(list(grads_batch_this_serialised))
            #model.mybuf=copy.deepcopy(tensor_to_send)
            
            #breakpoint()    
            
#            model.testbuf=torch.tensor(np.zeros(1))
            endtime=time.time()
            totaltime+=(endtime-starttime)
            #Tree Upward
#           for i in range(int(math.log2(size))):
#           for i in range(len(btreedata)):
            endtime=time.time()
            totaltime+=(endtime-starttime)
            for currentrow in btreedata1:
                         #logging.info(f"Rank,{rank},currentrow,{currentrow[0],currentrow[1]}")
                         if int(currentrow[0]) == rank:
                           dist.send(tensor=torch.tensor(len(tensor_to_send),dtype=torch.int64),dst=int(currentrow[1]))
                           dist.send(tensor=tensor_to_send,dst=int(currentrow[1]))
                           bytes_sent += len(tensor_to_send)
                           messages_sent += 1
                         elif int(currentrow[1]) == rank:
                           temp_tensor=torch.tensor(0,dtype=torch.int64)
                           dist.recv(tensor=temp_tensor,src=int(currentrow[0]))
                           model.mybuf=torch.ByteTensor(temp_tensor.item())
                           dist.recv(tensor=model.mybuf,src=int(currentrow[0]))
#                          param.grad.data+=model.mybuf
                           received_bytes = bytes(model.mybuf.tolist())
                           grads_batch_that=pickle.loads(received_bytes)
                           #cipherz = pickle.loads(received_bytes)
                           both_gradients=[]
                           both_gradients.append(grads_batch_this)
                           both_gradients.append(grads_batch_that)
                           starttime=time.time()
                           grads_batch_this = aggregate_gradients(both_gradients)
                           endtime=time.time()
                           totaltime+=(endtime-starttime)
                           grads_batch_this_serialised=pickle.dumps(grads_batch_this)
                           tensor_to_send = torch.ByteTensor(list(grads_batch_this_serialised))
                         
            if rank == root:
                 starttime=time.time()
                 grads_batch_final=unquantize_per_layer(grads_batch_this, r_maxs, bit_width=q_width)
                 endtime=time.time()
                 totaltime+=(endtime-starttime)
                 param.grad.data = torch.from_numpy(grads_batch_final)    

#Tree Downward
            model.mybuf=copy.deepcopy(param.grad.data)
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

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def run(rank, size, epochs, K, averager, runid, root):
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
    runstarttime=time.time()
    
    global nextadjustment,device
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
               if averager == "DFLBASICCQ":
                  bytes_sent,messages_sent=basic_average_gradients_cq(model,root)
                  total_bytes += bytes_sent
                  total_messgaes += messages_sent
               elif averager == "DFLBASICCQBEN":
                  bytes_sent,messages_sent=basic_average_gradients_cq_ben(model,root)                  
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



def init_processes(rank, size, epochs, K, averager, runid, root, fn, backend='gloo'):
   """ Initialize the distributed environment. """
   dist.init_process_group(backend, rank=rank, world_size=size)
   fn(rank, size, epochs, K, averager, runid, root)

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
    parser.add_argument("--root", type=int)
    args = parser.parse_args()
    rank = int(args.rank)
    size = int(args.size)
    epochs = int(args.epochs)
    averager = args.averager
    K = int(args.K)
    runid = args.runid
    root = int(args.root)
    init_processes(rank, size, epochs, K, averager, runid, root, run)
