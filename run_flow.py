import torch
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import get_batch_unin_dataset_withlabel

import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from PIL import Image
import os
import pandas as pd
import numpy as np
import inspect
from torchvision import transforms
from codebase import utils as ut
from codebase.models.mask_vae_flow import CausalVAE
import argparse
from pprint import pprint
from torchvision.utils import save_image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_max',   type=int, default=101,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="flow_mask",     help="Flag for toy")
args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x
    
class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[S?nderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n
	def __iter__(self):
		return self
	def __next__(self):
		t = self.t + self.inc
		self.t = self.t_max if t > self.t_max else t
		return self.t

layout = [
	('model={:s}',  'causalvae'),
	('run={:04d}', args.run),
	('color=True', args.color),
	('toy={:s}', str(args.toy))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
lvae = CausalVAE(name=model_name, z_dim=16).to(device)
if not os.path.exists('./figs_vae_flow/'):
	os.makedirs('./figs_vae_flow/')
def load_dag():
	flow_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
	df_np = np.loadtxt(flow_path+'/causal_data/causal_data/flow/1_0.8_3000/dag.txt')
	return df_np
dag = load_dag()
dataset_dir = './causal_data/causal_data/flow/1_0.8_3000'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 64)
test_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 1)
optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
dag = torch.as_tensor(torch.from_numpy(dag), dtype=torch.float32)
def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
for epoch in range(args.epoch_max):
	lvae.train()
	total_loss = 0
	total_rec = 0
	total_kl = 0
	for u, l in train_dataset:
		optimizer.zero_grad()
		u = u.to(device)
		L, kl, rec, reconstructed_image,_ = lvae.negative_elbo_bound(u, l, dag, sample=False)
		L.backward()
		optimizer.step()
		total_loss += L.item()
		total_kl += kl.item() 
		total_rec += rec.item()
		m = len(train_dataset)
		save_image(u, 'figs_vae_flow/reconstructed_image_true_{}.png'.format(epoch), normalize=True)
		save_image(reconstructed_image, 'figs_vae_flow/reconstructed_image_{}.png'.format(epoch), range=(0, 1), normalize=True)
	if epoch % 1 == 0:
		print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))
	if epoch % args.iter_save == 0:
		ut.save_model_by_name(lvae, epoch)

