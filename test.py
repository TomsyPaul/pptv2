import pickle
import os, sys
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

def run(rank, size):
    """ Distributed Synchronous SGD Example """
#    torch.manual_seed(1234)
#    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    publickey=paillier.PaillierPublicKey(27236700922646976555595848507913589494886491119135730116014077185311243444450372255376489388880173022641848729747213088746475118480344996406749938547224285029951417411158327330610634671230458266993515963753271442282969744291116368707834837036890519842176076657317424175485854349519237230877898852294685281803161775833139254050216610420167131637216465657783550454961204111753470621658424459969937833601118914414496472033175054121693273513687334787107976849759736841476647931918984474457173711208172669939800415050356154977238127550304510079658979903408556459392897794799075517038480041829170731623511642064703877042081)
    privatekey=paillier.PaillierPrivateKey(publickey,151731466574845229173039099025495922095008727037737139342593172671294559941585338353476978224816381692776568780654141379462769869960946855764863957672293447968060874024631012137102252157537274255768018381630660935684213060267736952413889789815316743297286676968944107787802792481731771876027538148701189447527,179505949144779488794639217468096838709557457111534772428337237788171867936597162601187903575540819127792895820899862337184627927909110508331699702369829930039159510893967892328575714125966954175352601542034250713387089395862180071677366675266103958356045025432474981187150877764707656153946356889833385368503)
#    gradients=[]
    bytes_sent=0
    if rank == 0:
         x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
         npx=x.numpy()
         cipherx=encryption.encrypt_matrix(publickey, npx)
         serialized_cipherx = pickle.dumps(cipherx)
         tensor_to_send = torch.ByteTensor(list(serialized_cipherx))
         dist.send(tensor=tensor_to_send,dst=1)
         bytes_sent += tensor_to_send.nelement() * tensor_to_send.element_size()
         print("In 0.. Sent..x=",x,"\nCipherx=",cipherx,"\nBytes Sent..",bytes_sent)
    else:
         y = torch.tensor([[7, 4, 3], [5, 7, 1]], dtype=torch.float32)
         npy=y.numpy()
         ciphery=encryption.encrypt_matrix(publickey, npy)
         serialized_ciphery = pickle.dumps(ciphery)
         byte_tensor_y=torch.ByteTensor(list(serialized_ciphery))
         received_tensor=copy.deepcopy(byte_tensor_y)
         dist.recv(tensor=received_tensor,src=0)
         received_bytes = bytes(received_tensor.tolist())
         cipherz = pickle.loads(received_bytes)
         print("In 1.. Recvd..",cipherz)
         zarray=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in cipherz])    
#         print("Dec Result..\ncipherz= ",cipherz,"zarray=  ", zarray)
         z=torch.from_numpy(zarray)
         print("z=",z)
         test_list=[]
         test_list.append(ciphery)
         test_list.append(cipherz)
         cipherresult=aggregate_gradients(test_list)
         result=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in cipherresult])
         print("\nSum array = ",result)
         resulttensor=torch.from_numpy(result)
         print("\nSum tensor = ",resulttensor)
    #opposite.. x=torch.from_numpy(npx)
    
    
    #print("Decrypring..")
    #breakpoint()
    #backtosum=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in ciphersum])
    #print("Dec Result..\nciphersum= ",ciphersum,"backtosum=  ", backtosum)
    #gradsum=torch.from_numpy(backtosum)
    #print("gradsum=",gradsum)

#    z=torch.zeros_like(x)







def init_processes(rank, size,fn, backend='gloo'):
   """ Initialize the distributed environment. """
   dist.init_process_group(backend, rank=rank, world_size=size)
   fn(rank, size)

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
    args = parser.parse_args()
    rank = int(args.rank)
    size = int(args.size)
    init_processes(rank, size, run)
