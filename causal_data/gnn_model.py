import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn import GNN_edge
from utils.utils import get_activation

def get_gnn(data, args):
    model = GNN(data.num_node_features, data.train_edge_value_dim,
                args.node_dim, args.edge_dim,
                args.model_types, args.dropout, args.gnn_activation,
                args.concat_states, args.post_hiddens,
                args.norm_embs,)
    return model

class GNN(torch.nn.Module):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs,):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types

        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim,
                                    normalize_embs, activation)
        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_dim, activation)
        self.node_post_mlp = self.build_node_post_mlp(node_dim, node_post_mlp_hiddens, dropout, activation)

    def build_node_post_mlp(self, input_dim, hidden_dims, dropout, activation):
        layers = []
        input_dims = int(input_dim / 2)
        layer = nn.Sequential(
                    nn.Linear(input_dims, hidden_dims),
                    get_activation(activation),
                    nn.Dropout(dropout),
                    )
        layers.append(layer)
        #layer = nn.Linear(hidden_dims, input_dims)
        layer = nn.Linear(hidden_dims, int(hidden_dims/2))
        layers.append(layer)
        return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim,
                     normalize_embs, activation):
        convs = nn.ModuleList()
        output_dim = int(node_dim / 2)
        conv = self.build_conv_model(node_input_dim,node_dim,
                                    edge_input_dim, normalize_embs, activation,)
        convs.append(conv)
        conv = self.build_conv_model(node_dim, node_dim,
                                    edge_dim, normalize_embs, activation,)
        convs.append(conv)
        conv = self.build_conv_model(node_dim, output_dim,
                                    edge_dim, normalize_embs, activation, )
        convs.append(conv)

        return convs

    def build_conv_model(self, node_in_dim, node_out_dim, edge_dim, normalize_emb, activation):

        return GNN_edge(node_in_dim, node_out_dim, edge_dim, activation, normalize_emb)

    def build_edge_update_mlps(self, node_dim, edge_dim, activation):
        edge_update_mlps = nn.ModuleList()
        output_dim = int(edge_dim / 2)
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim, edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        edge_update_mlp = nn.Sequential(
            nn.Linear(node_dim+node_dim, edge_dim),
            get_activation(activation),
            )
        edge_update_mlps.append(edge_update_mlp)
        edge_update_mlp = nn.Sequential(
            nn.Linear(node_dim, output_dim),
            get_activation(activation),
        )
        edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_value = mlp(torch.cat((x_i, x_j), dim=-1))
        return edge_value

    def update_x(self, x, edge_value, edge_index, conv):
        x = conv(x, edge_value, edge_index)
        return x

    def forward(self, x, edge_value, edge_index):
        conv = self.convs
        edge_update_mlp = self.edge_update_mlps
        for l in range(0,3):
            x = self.update_x(x, edge_value, edge_index, conv[l])
            edge_value = self.update_edge_attr(x, edge_index, edge_update_mlp[l])
        x = self.node_post_mlp(x)
        return x
