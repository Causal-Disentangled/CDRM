import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing
from utils.data_utils import get_known_mask, create_graph, create_matrix, create_edge, create_edge_value

import torch
import random
import numpy as np


def get_data(df_X, df_Y, df_Z,normalize=True):
    df_Y = df_Y.to_numpy()
    df_Z = df_Z.to_numpy()
    row, col = df_X.shape
    node_num = df_Y.size

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df_X)
        df_X = pd.DataFrame(x_scaled)

    edge_start, edge_end = create_edge(row, col)
    edge_index = torch.tensor([edge_start, edge_end], dtype=torch.int64)
    edge_value = create_edge_value(df_X, row, col)
    edge_value = torch.tensor(edge_value, dtype=torch.float)

    matrix_init = create_matrix(row, col)#

    x = torch.tensor(matrix_init, dtype=torch.float)

    known_mask = get_known_mask(df_Z, int(node_num))
    double_known_edge_mask = torch.cat((known_mask, known_mask), dim=0)

    train_edge_index, train_edge_value = create_graph(edge_index, edge_value, double_known_edge_mask)#know
    train_value = train_edge_value[:int(train_edge_value.shape[0] / 2), 0]
    train_edge_value_dim = train_edge_value.shape[-1]

    test_edge_index, test_edge_value = create_graph(edge_index, edge_value, ~double_known_edge_mask)#missing
    test_value = test_edge_value[:int(test_edge_value.shape[0] / 2), 0]

    data = Data(x=x, edge_index=edge_index, edge_value=edge_value,
                train_edge_index=train_edge_index, train_edge_value=train_edge_value,
                train_value=train_value, test_value=test_value,
                test_edge_index=test_edge_index, test_edge_value=test_edge_value,
                df_X=df_X, df_y=df_Y, df_Z=df_Z, min_max_scaler=min_max_scaler,
                train_edge_value_dim=train_edge_value_dim, row=row, col=col,
                )
    return data


def load_data():
    flow_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(flow_path+'\\causal_data\\original_flow\\flow.txt')
    df_X = pd.DataFrame(df_np[:, :-2])  #data
    df_Y = pd.DataFrame(df_np[:, -2])  #label
    df_Z = pd.DataFrame(df_np[:, -1:])  #missing
    data = get_data(df_X, df_Y, df_Z)
    return data
