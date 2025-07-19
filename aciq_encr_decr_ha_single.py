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

from joblib import Parallel, delayed
import multiprocessing
N_JOBS = multiprocessing.cpu_count()

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

from functools import reduce
from ftl.encryption import paillier, encryption

def clip_gradients(grads, min_v, max_v):
    results = [torch.clip(t, min=min_v, max=max_v).numpy() for t in grads]
    return results

def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results

def aggregate_gradients(gradient_list, weight=0.5):
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


if  __name__ == "__main__":
    publickey=paillier.PaillierPublicKey(27236700922646976555595848507913589494886491119135730116014077185311243444450372255376489388880173022641848729747213088746475118480344996406749938547224285029951417411158327330610634671230458266993515963753271442282969744291116368707834837036890519842176076657317424175485854349519237230877898852294685281803161775833139254050216610420167131637216465657783550454961204111753470621658424459969937833601118914414496472033175054121693273513687334787107976849759736841476647931918984474457173711208172669939800415050356154977238127550304510079658979903408556459392897794799075517038480041829170731623511642064703877042081)
    privatekey=paillier.PaillierPrivateKey(publickey,151731466574845229173039099025495922095008727037737139342593172671294559941585338353476978224816381692776568780654141379462769869960946855764863957672293447968060874024631012137102252157537274255768018381630660935684213060267736952413889789815316743297286676968944107787802792481731771876027538148701189447527,179505949144779488794639217468096838709557457111534772428337237788171867936597162601187903575540819127792895820899862337184627927909110508331699702369829930039159510893967892328575714125966954175352601542034250713387089395862180071677366675266103958356045025432474981187150877764707656153946356889833385368503)
    bytes_sent=0
    x = torch.tensor([[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16], [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]], dtype=torch.float32)
    npx=x.numpy()
    cipherx=encryption.encrypt_matrix(publickey, npx)
    y = torch.tensor([[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]], dtype=torch.float32)
    npy=y.numpy()
    ciphery=encryption.encrypt_matrix(publickey, npy)
    # test_list=[]
    # test_list.append(cipherx)
    # test_list.append(ciphery)
    # cipherresult=aggregate_gradients(test_list)
    # result=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in cipherresult])
    # print("\nx array = ",x)
    # print("\ny array = ",y)
    # print("\nSum array = ",result)
    # resulttensor=torch.from_numpy(result)
    #print("\nSum tensor = ",resulttensor)
    num_clients=2
    grads_batch_clients=[x.numpy(),y.numpy()]
    q_width=16
    sizes = [item.size * num_clients for item in grads_batch_clients[0]]
    max_values = []
    min_values = []
    for layer_idx in range(len(grads_batch_clients[0])):
        max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
        min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
    grads_max_min = np.concatenate([np.array(max_values),np.array(min_values)],axis=1)
    clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)


    r_maxs = [x * num_clients for x in clipping_thresholds]

    # grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
    # grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
    grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                            for item in grads_batch_clients]

    # grads_0 = quantize_per_layer(grads_0, r_maxs, bit_width=q_width)
    # grads_1 = quantize_per_layer(grads_1, r_maxs, bit_width=q_width)
    grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width)
                            for item in grads_batch_clients]
    #breakpoint()
    # grads = aggregate_gradients([grads_0, grads_1])
    # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
    grads = aggregate_gradients(grads_batch_clients)
    client_weight = 1.0 / num_clients
    #loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

    # grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)
    grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)
    #result=np.array([encryption.decrypt_matrix(privatekey, item).astype(np.float32) for item in grads])
    print("\nx array = ",x)
    print("\ny array = ",y)
    print("\nSum array = ",grads)



    print("\n\n\nStarting en_batch ") 
    theta = 2.5
    # clipping_thresholds = encryption.calculate_clip_threshold(grads_0) return [theta * np.std(x) for x in grads]
    # theta = 2.5
    # calculate global std by combination, clients send E(X^2), E(X), and layerwise sizes to the server
    # std = E(X^2) - (E(X))^2
    num_clients=2
    grads_batch_clients=[x.numpy(),y.numpy()]
    q_width=16
    sizes = [item.size * num_clients for item in grads_batch_clients[0]]

    grads_batch_clients_mean = []
    grads_batch_clients_mean_square = []
    for client_idx in range(len(grads_batch_clients)):
        temp_mean = [np.mean(grads_batch_clients[client_idx][layer_idx])
                        for layer_idx in range(len(grads_batch_clients[client_idx]))]
        temp_mean_square = [np.mean(grads_batch_clients[client_idx][layer_idx] ** 2)
                            for layer_idx in range(len(grads_batch_clients[client_idx]))]
        grads_batch_clients_mean.append(temp_mean)
        grads_batch_clients_mean_square.append(temp_mean_square)
    grads_batch_clients_mean = np.array(grads_batch_clients_mean)
    grads_batch_clients_mean_square = np.array(grads_batch_clients_mean_square)

    layers_size = np.array([_.size for _ in grads_batch_clients[0]])
    clipping_thresholds = theta * (
                np.sum(grads_batch_clients_mean_square * layers_size, 0) / (layers_size * num_clients)
                - (np.sum(grads_batch_clients_mean * layers_size, 0) / (layers_size * num_clients)) ** 2) ** 0.5

    print("clipping_thresholds", clipping_thresholds)

    r_maxs = [x * num_clients for x in clipping_thresholds]

    # grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
    # grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
    grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                            for item in grads_batch_clients]

    # enc_grads_0, og_shape_0 = batch_enc_per_layer(publickey=publickey, party=grads_0, r_maxs=r_maxs, bit_width=q_width,
    #                                               batch_size=batch_size)
    # enc_grads_1, og_shape_1 = batch_enc_per_layer(publickey=publickey, party=grads_1, r_maxs=r_maxs, bit_width=q_width,
    #                                               batch_size=batch_size)
    enc_grads_batch_clients = []
    og_shape_batch_clients = []
    batchsize=16
    for item in grads_batch_clients:
        enc_grads_temp, og_shape_temp = batch_enc_per_layer(publickey=publickey, party=item,
                                                            r_maxs=r_maxs,
                                                            bit_width=q_width,
                                                            batch_size=batchsize)
        enc_grads_batch_clients.append(enc_grads_temp)
        og_shape_batch_clients.append(og_shape_temp)

    # loss_value_0 = publickey.encrypt(loss_value_0)
    # loss_value_1 = publickey.encrypt(loss_value_1)
    #loss_batch_clients = [publickey.encrypt(item) for item in loss_batch_clients]

    # grads = aggregate_gradients([enc_grads_0, enc_grads_1])
    # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
    grads = aggregate_gradients(enc_grads_batch_clients)
    serialized_grads = pickle.dumps(grads)
    #client_weight = 1.0 / num_clients
    #loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

    # loss_value = encryption.decrypt(privatekey, loss_value)
    # grads = batch_dec_per_layer(privatekey=privatekey, party=grads, og_shapes=og_shape_0, r_maxs=r_maxs, bit_width=q_width, batch_size=batch_size)
    #loss_value = encryption.decrypt(privatekey, loss_value)
    grads = batch_dec_per_layer(privatekey=privatekey, party=grads, og_shapes=og_shape_batch_clients[0],
                                r_maxs=r_maxs, bit_width=q_width, batch_size=batchsize)
    print("\n\nResult= ", grads)
    print("\n\nSize= ", len(serialized_grads))
