import time
import argparse
import sys
import os

import numpy as np
import torch
import pandas as pd

from flow_data import load_data
from gnn_flow import train_gnn_flow
from scm_flow import dag
from restore_flow import restore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE')
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default='True')
    parser.add_argument('--post_hiddens', type=str, default=32)
    parser.add_argument('--node_dim', type=int, default=128)
    parser.add_argument('--edge_dim', type=int, default=128)
    parser.add_argument('--impute_hiddens', type=str, default=128)
    parser.add_argument('--pre_hidden_dim', type=str, default=64)
    parser.add_argument('--output_dim', type=str, default=1)
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--output_activation', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--ratio', type=int, default=0.8)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_dir', type=str, default='1')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    data = load_data()

    if not os.path.exists('./causal_data/flow/'):
        os.makedirs('./causal_data/flow/')
    log_path = './causal_data/flow/{}_{}_{}/'.format(args.log_dir, args.ratio, args.epochs)
    os.makedirs(log_path)
    train_gnn_flow(data, args, log_path, device)
    dag(args)
    restore(args)

    #print(scm)


if __name__ == '__main__':
    main()