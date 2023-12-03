import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import numpy as np
import os.path as osp
import inspect
import pandas as pd

def projection(theta, phi, x, y, base=-0.5):
    b = y - x * math.tan(phi)
    shade = (base - b) / math.tan(phi)
    return shade

def restore(args):

    flow_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(flow_path+'/causal_data/pendulum/{}_{}_{}/restore_pendulum_{}.txt'.format(args.log_dir, args.ratio, args.epochs, args.epochs))
    df_X = pd.DataFrame(df_np[:, :])  #数据
    row, col = df_X.shape

    os.makedirs('./causal_data/pendulum/{}_{}_{}/train/'.format(args.log_dir, args.ratio, args.epochs))
    os.makedirs('./causal_data/pendulum/{}_{}_{}/test/'.format(args.log_dir, args.ratio, args.epochs))
    count = 0
    a = 0
    flow_dir = ('./causal_data/pendulum/{}/'.format(args.log_dir))
    fname = 'pendulum'
    path = os.path.join(flow_dir, fname + '.txt')

    scale = np.array([[0, 44], [100, 40], [7, 7.5], [10, 10]])
    empty = pd.DataFrame(columns=['i', 'j', 'shade', 'mid'])

    for i in range(row):
        data = [df_X[0][i], df_X[1][i], df_X[2][i], df_X[3][i]]
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        theta = data[0] * math.pi / 200.0
        phi = data[1] * math.pi / 200.0
        x = 10 + 8 * math.sin(theta)
        y = 10.5 - 8 * math.cos(theta)

        ball = plt.Circle((x, y), 1.5, color='firebrick')
        gun = plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3)

        light = projection(theta, phi, 10, 10.5, 20.5)
        sun = plt.Circle((light, 20.5), 3, color='orange')

        shadow = plt.Polygon(([data[3] - data[2] / 2.0, -0.5], [data[3] + data[2] / 2.0, -0.5]), color='black', linewidth=3)

        ax = plt.gca()
        ax.add_artist(gun)
        ax.add_artist(ball)
        ax.add_artist(sun)
        ax.add_artist(shadow)
        ax.set_xlim((0, 20))
        ax.set_ylim((-1, 21))
        new = pd.DataFrame({
            'i': (data[0] - scale[0][0]) / (scale[0][1] - 0),
            'j': (data[1] - scale[1][0]) / (scale[1][1] - 0),
            'shade': (data[3] - scale[2][0]) / (scale[2][1] - 0),
            'mid': (data[2] - scale[2][0]) / (scale[2][1] - 0)
        },

            index=[1])
        empty = empty.append(new, ignore_index=True)
        plt.axis('off')

        a = a + 1
        if count == 4:
            plt.savefig('./causal_data/pendulum/{}_{}_{}/test/a_'.format(args.log_dir, args.ratio, args.epochs) + str(int(data[0])) + "_" + str(int(data[1])) + "_" + str(int(data[2])) + "_" + str(int(data[3])) + "_" + str(a) + '.png',dpi=96)
            count = 0
        else:
            plt.savefig('./causal_data/pendulum/{}_{}_{}/train/a_'.format(args.log_dir, args.ratio, args.epochs) + str(int(data[0])) + "_" + str(int(data[1])) + "_" + str(int(data[2])) + "_" + str(int(data[3])) + "_" + str(a) + '.png',dpi=96)
            count += 1

        plt.clf()
        # if count == 4:
        # plt.savefig('./causal_data/pendulum/test/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
        # count = 0
        # else:
        # plt.savefig('./causal_data/pendulum/train/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
        # plt.clf()
        # count += 1



