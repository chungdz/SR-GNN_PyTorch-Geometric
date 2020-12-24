# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
import argparse
import json
import pickle
from tqdm import tqdm
import time
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from torch_geometric.data import DataLoader
import math
from datasets.graph_datasets import MultiSessionsGraph
from datasets.gather import gather as gather_all
from src.model import GNNModel
from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric


def run(cfg: DictConfig, rank: int, device: torch.device, finished: mp.Value, train_dataset, valid_dataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # # Build model.
    model = GNNModel(hidden_size=cfg.hidden_size, n_node=cfg.n_node)
    model.to(device)

    # Build optimizer.
    steps_one_epoch = len(train_data_loader) // cfg.accu
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_dc_step, gamma=cfg.lr_dc)

    print("Worker %d is working ... " % rank)
    # Fast check the validation process
    if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
        validate(cfg, -1, model, device, rank, valid_data_loader, fast_dev=True)
        logging.warning(model)
        gather_all(cfg, 1)
    
    # Training and validation
    for epoch in range(cfg.epoch):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, rank, model, train_data_loader,
              optimizer, steps_one_epoch, device)
    
        validate(cfg, epoch, model, device, rank, valid_data_loader)
        # add finished count
        finished.value += 1

        if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
            save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.checkpoint_path)

            while finished.value < cfg.gpus:
                time.sleep(1)
            gather_all(cfg, cfg.gpus)
            finished.value = 0
        
        scheduler.step()


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def train(cfg, epoch, rank, model, loader, optimizer, steps_one_epoch, device):
    """
    train loop
    :param args: config
    :param epoch: int, the epoch number
    :param gpu_id: int, the gpu id
    :param rank: int, the process rank, equal to gpu_id in this code.
    :param model: gating_model.Model
    :param loader: train data loader.
    :param criterion: loss function
    :param optimizer:
    :param steps_one_epoch: the number of iterations in one epoch
    :return:
    """
    model.train()

    model.zero_grad()

    enum_dataloader = enumerate(loader)
    if ((cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0)):
        enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch * cfg.accu:
            break
        # data = {key: value.to(device) for key, value in data.items()}
        data = data.to(device)
        # 1. Forward
        scores = model(data)
        loss = model.loss_function(scores, data.y)

        if cfg.accu > 1:
            loss = loss / cfg.accu

        # 3.Backward.
        loss.backward()

        if (i + 1) % cfg.accu == 0:
            if cfg.gpus > 1:
                average_gradients(model)
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()

    # if (not args.dist_train) or (args.dist_train and rank == 0):
    #     util.save_checkpoint_by_epoch(
    #         model.state_dict(), epoch, args.checkpoint_path)


def validate(cfg, epoch, model, device, rank, valid_data_loader, fast_dev=False, top_k=20):
    model.eval()

    # Setting the tqdm progress bar
    if rank == 0:
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test:%d" % epoch,
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)
                        
    with torch.no_grad():
        hit, mrr = [], []
        for i, data in data_iter:
            if fast_dev and i > 10:
                break

            scores = model(data.to(device))
            targets = data.y

            sub_scores = scores.topk(top_k)[1]
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        tmp_dict = {}
        tmp_dict['hit'] = hit
        tmp_dict['mrr'] = mrr

        with open(cfg.result_path + 'tmp_{}.pkl'.format(rank), 'wb') as f:
            pickle.dump(tmp_dict, f)
        f.close()


def init_processes(cfg, local_rank, vocab, dataset, valid_dataset, finished, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = cfg.port
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, finished, train_dataset=dataset, valid_dataset=valid_dataset)


def split_dataset(dataset, gpu_count):
    sub_len = len(dataset) // gpu_count
    if len(dataset) != sub_len * gpu_count:
        len_a, len_b = sub_len * gpu_count, len(dataset) - sub_len * gpu_count
        dataset, _ = torch.utils.data.random_split(dataset, [len_a, len_b])

    return torch.utils.data.random_split(dataset, [sub_len, ] * gpu_count)

def split_valid_dataset(dataset, gpu_count):
    sub_len = math.ceil(len(dataset) / gpu_count)
    data_list = []
    for i in range(gpu_count):
        s = i * sub_len
        e = (i + 1) * sub_len
        data_list.append(dataset[s: e])

    return data_list

def main(cfg):
    
    set_seed(7)
    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + cfg.dataset, phrase='train')
    validate_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + cfg.dataset, phrase='test')
    item_dict = json.load(open(cur_dir + '/datasets/' + cfg.dataset + '/item_dict.json', 'r', encoding='utf-8'))
    cfg.n_node = len(item_dict)
    cfg.result_path = cur_dir + '/datasets/' + cfg.dataset + '/result/'
    cfg.checkpoint_path = cur_dir + '/datasets/' + cfg.dataset + '/checkpoint/'
    finished = mp.Value('i', 0)

    assert(cfg.gpus > 1)
    dataset_list = split_dataset(train_dataset, cfg.gpus)
    valid_dataset_list = split_valid_dataset(validate_dataset, cfg.gpus)

    processes = []
    for rank in range(cfg.gpus):
        p = mp.Process(target=init_processes, args=(
            cfg, rank, None, dataset_list[rank], valid_dataset_list[rank], finished, run, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--gpus', type=int, default=2, help='gpu_num')
    parser.add_argument('--accu', type=int, default=1, help='number of train time')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
    parser.add_argument('--port', type=int, default=9337)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
