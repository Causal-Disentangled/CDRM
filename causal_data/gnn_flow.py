import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import os
import random

from gnn_model import get_gnn
from prediction_model import get_MLP
from utils.plot_utils import plot_curve
from utils.data_utils import get_train_mask, create_graph

def write_data(df_X, args):
    data_dir = './causal_data/flow/{}_{}_{}/'.format(args.log_dir, args.ratio, args.epochs)
    fname = 'restore_flow_{}'.format(args.epochs)
    data_path = os.path.join(data_dir, fname + '.txt')
    row, col = df_X.shape
    for i in range(row):
        a = float(df_X.iloc[i, 0])
        b = float(df_X.iloc[i, 1])
        c = float(df_X.iloc[i, 2])
        d = float(df_X.iloc[i, 3])
        with open(data_path, 'a') as f:
            f.write(str(int(a))+" "+str(int(b))+" "+str(int(c))+" "+str(int(d))+'\n')

def train_gnn_flow(data, args, log_path, device=torch.device('cpu')):
    train_model = get_gnn(data, args).to(device)
    pre_model = get_MLP(args).to(device)
    df_Z = data.df_Z
    min_max_scaler = data.min_max_scaler

    opt = torch.optim.Adam(list(train_model.parameters()) + list(pre_model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    Train_loss = []
    Train_rmse = []
    Test_rmse = []
    Test_l1 = []

    x = data.x.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_value = data.train_edge_value.clone().detach().to(device)
    train_value = data.train_value.clone().detach().to(device)
    test_edge_index = data.test_edge_index.clone().detach().to(device)
    test_edge_value = data.test_edge_value.clone().detach().to(device)
    test_value = data.test_value.clone().detach().to(device)
    row = data.row
    col = data.col
    ratio = args.ratio

    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()
    for epoch in range(args.epochs):
        train_model.train()
        pre_model.train()
        opt.zero_grad()

        train_mask = get_train_mask(ratio, int(train_edge_value.shape[0] / 2))
        double_train_mask = torch.cat((train_mask, train_mask), dim=0)
        known_edge_index, known_edge_value = create_graph(train_edge_index, train_edge_value,
                                                       double_train_mask)

        train_x_emb = train_model(x, known_edge_value, known_edge_index)
        train_pre = pre_model([train_x_emb[train_edge_index[0]], train_x_emb[train_edge_index[1]]])
        train_pre_value = train_pre[:int(train_edge_value.shape[0] / 2), 0]

        loss = F.mse_loss(train_pre_value, train_value)
        model_loss = loss.item()
        train_rmse = np.sqrt(loss.item())
        loss.backward()
        opt.step()

        train_model.eval()
        pre_model.eval()
        with torch.no_grad():  # test
            test_x_emb = train_model(x, train_edge_value, train_edge_index)
            test_pre = pre_model([test_x_emb[test_edge_index[0]], train_x_emb[test_edge_index[1]]])
            test_pre_value = test_pre[:int(test_edge_value.shape[0] / 2), 0]

            mse = F.mse_loss(test_pre_value, test_value)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(test_pre_value, test_value)
            test_l1 = l1.item()

            Train_loss.append(model_loss)
            Train_rmse.append(train_rmse)
            Test_rmse.append(test_rmse)
            Test_l1.append(test_l1)

            print('epoch: ', epoch)
            print('loss: ', model_loss)
            print('train rmse: ', train_rmse)
            print('test rmse: ', test_rmse)
            print('test l1: ', test_l1)

    pred_num_val = test_pre_value.detach().cpu().numpy()
    train_num_val = train_value.detach().cpu().numpy()

    restore_data = []
    n = 0
    m = 0
    for i in range(row):
        for j in range(col):
            if j != df_Z[i]:
                restore_data.append(train_num_val[n])
                n = n + 1
            else:
                restore_data.append(pred_num_val[m])
                m = m + 1

    restore_pre_num_val = np.array(restore_data).reshape(-1, 4)
    restore_data = min_max_scaler.inverse_transform(restore_pre_num_val)
    restore_data = np.round(restore_data)
    df_a = pd.DataFrame(restore_data)
    write_data(df_a, args)

    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    obj['curves']['train_rmse'] = Train_rmse
    obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_l1'] = Test_l1

    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    if args.save_model:
        torch.save(train_model, log_path + 'model.pt')
        torch.save(pre_model, log_path + 'impute_model.pt')

    plot_curve(obj['curves'], log_path+'curves.png', keys=None,
                clip=True, label_min=True, label_end=True)
