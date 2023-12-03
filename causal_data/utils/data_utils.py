import torch
import random
import numpy as np


def create_edge(row, col):
    start_edge = []
    end_edge = []
    for x in range(row):
        start_edge = start_edge + [x] * col
        end_edge = end_edge + list(row + np.arange(col))
    double_start_edge = start_edge + end_edge
    double_end_edge = end_edge + start_edge
    return double_start_edge, double_end_edge


def create_edge_value(df_x, row, col):
    edge_value = []
    for i in range(row):
        for j in range(col):
            edge_value.append([float(df_x.iloc[i, j])])
    edge_value = edge_value + edge_value
    return edge_value


def create_matrix(row, col):
    feature_ind = np.array(range(col))
    feature_node = np.zeros((col, col))
    feature_node[np.arange(col), feature_ind] = 1
    sample_node = [[1] * col for i in range(row)]
    node = sample_node + feature_node.tolist()
    return node


def get_known_mask(df_Z, node_num):
    known_mask = []
    for i in range(node_num):
        mask = [True for i in range(4)]
        if df_Z[i] == 0:
            mask[0] = False
        if df_Z[i] == 1:
            mask[1] = False
        if df_Z[i] == 2:
            mask[2] = False
        if df_Z[i] == 3:
            mask[3] = False
        known_mask = known_mask + mask
    known_mask = (torch.tensor(known_mask).view(-1))
    return known_mask


def create_graph(edge_index, edge_value, mask):
    edge_index = edge_index.clone().detach()
    edge_value = edge_value.clone().detach()
    edge_index = edge_index[:,mask]
    edge_value = edge_value[mask]
    return edge_index, edge_value


def get_train_mask(ratio, edge_num):
    train_mask = (torch.FloatTensor(edge_num, 1).uniform_() < ratio).view(-1)
    return train_mask

def create_onehot_matrix(row, col):
    feature_ind = np.array(range(col))
    feature_node = np.zeros((col, col + 1))
    feature_node[np.arange(col), feature_ind + 1] = 1
    sample_node = np.zeros((row, col + 1))
    sample_node[:, 0] = 1
    node = sample_node.tolist() + feature_node.tolist()
    return node