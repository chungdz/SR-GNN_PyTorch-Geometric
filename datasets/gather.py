# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
import argparse
from tqdm import tqdm
import pickle
import scipy.stats as ss
import numpy as np
import pandas as pd
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

def gather(cfg, filenum):
    output_path = cfg.result_path

    hit, mrr = [], []

    for i in range(filenum):
        with open(output_path + 'tmp_{}.pkl'.format(i), 'rb') as f:
            cur_result = pickle.load(f)
        hit += cur_result['hit']
        mrr += cur_result['mrr']
    
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    print('index/hit', hit)
    print('index/mrr', mrr)


    

