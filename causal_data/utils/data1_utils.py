import torch
import random
import numpy as np

def get_known_mask(df_Z, node_num):
    known_mask = []
    for i in range(node_num):
        if df_Z[i] == 0:
            known_mask = known_mask + [False,True,True,True]
        if df_Z[i] == 1:
            known_mask = known_mask + [True,False,True,True]
        if df_Z[i] == 2:
            known_mask = known_mask + [True,True,False,True]
        if df_Z[i] == 3:
            known_mask = known_mask + [True,True,True,False]
    known_mask = (torch.tensor(known_mask).view(-1))
    #print(known_mask)
    return known_mask

def mask_edge(edge_index, edge_value, mask):
    edge_index = edge_index.clone().detach()
    edge_value = edge_value.clone().detach()
    edge_index = edge_index[:,mask]
    edge_value = edge_value[mask]
    return edge_index, edge_value

def create_matrix(df_X):
    row, col = df_X.shape
    feature_ind = np.array(range(col))
    feature_node = np.zeros((col, col))
    feature_node[np.arange(col), feature_ind] = 1
    sample_node = [[1] * col for i in range(row)]
    node = sample_node + feature_node.tolist()
    return node

def create_edge(df_X):
    row, col = df_X.shape
    start_edge = []
    end_edge = []
    for x in range(row):
        start_edge = start_edge + [x] * col
        end_edge = end_edge + list(row + np.arange(col))
    double_start_edge = start_edge + end_edge
    double_end_edge = end_edge + start_edge
    return double_start_edge, double_end_edge

def create_edge_value(df_X):
    row, col = df_X.shape
    edge_value = []
    for i in range(row):
        for j in range(col):
            edge_value.append([float(df_X.iloc[i, j])])
    edge_value = edge_value + edge_value
    return edge_value