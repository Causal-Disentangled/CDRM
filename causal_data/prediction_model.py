import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation

def get_MLP(args):
	model = MLP(args.node_dim, args.pre_hidden_dim, args.output_dim,
				args.impute_activation, args.output_activation,
				args.dropout,)
	return model

class MLP(torch.nn.Module):
	def __init__(self,
				input_dims, hidden_dim, output_dim,
				activation,output_activation,
				dropout):
		super(MLP, self).__init__()
		self.activation = activation

		layers = nn.ModuleList()
		input_dims = int(input_dims / 4)
		hidden_dim = int(hidden_dim / 4)
		update_hidden_dim = int(hidden_dim / 2)
		layer = nn.Sequential(
					nn.Linear(input_dims, hidden_dim),
					get_activation(activation),
					nn.Dropout(dropout),
					)
		layers.append(layer)
		layer = nn.Sequential(
					nn.Linear(hidden_dim, update_hidden_dim),
					get_activation(activation),
					nn.Dropout(dropout),
					)
		layers.append(layer)
		layer = nn.Sequential(
						nn.Linear(update_hidden_dim, output_dim),
						get_activation(output_activation),
						)
		layers.append(layer)
		self.layers = layers

	def update_input_var(self, input_var, layer):
		input_var = layer(input_var)
		return input_var

	def forward(self, inputs):
		input_var = torch.cat(inputs,-1)
		layer = self.layers
		for i in range(0,3):
			input_var = self.update_input_var(input_var, layer[i])
		return input_var
