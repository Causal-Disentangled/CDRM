import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd
import os.path as osp
import inspect
import argparse

def restore(args):
    count = 0
    i = 0

    flow_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(flow_path+'/causal_data/flow/{}_{}_{}/restore_flow_{}.txt'.format(args.log_dir, args.ratio, args.epochs, args.epochs))
    df_X = pd.DataFrame(df_np[:, :])  #data
    row, col = df_X.shape

    os.makedirs('./causal_data/flow/{}_{}_{}/train/'.format(args.log_dir, args.ratio, args.epochs))
    os.makedirs('./causal_data/flow/{}_{}_{}/test/'.format(args.log_dir, args.ratio, args.epochs))

    for i in range(row):
        data = [df_X[0][i], df_X[1][i], df_X[2][i], df_X[3][i]]
        ball_r = data[0] / 30.0
        h = data[1]
        deep = data[3] / 3.0
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        ax = plt.gca()

        # water in cup
        rect = plt.Rectangle(([3, 0]), 5, 5 + h, color='lightskyblue')
        ax.add_artist(rect)
        ball = plt.Circle((5.5, +ball_r + 0.5), ball_r, color='firebrick')

        ## cup
        left = plt.Polygon(([3, 0], [3, 19]), color='black', linewidth=2)
        right_1 = plt.Polygon(([8, 0], [8, deep]), color='black', linewidth=2)
        right_2 = plt.Polygon(([8, deep + 0.4], [8, 19]), color='black', linewidth=2)
        ax.add_artist(left)
        ax.add_artist(right_1)
        ax.add_artist(right_2)
        ax.add_artist(ball)

        # water line
        y = np.linspace(deep, 0.5)
        epsilon = 0.01 * np.max([np.abs(np.random.randn(1)), 1])
        x = np.sqrt(2 * (0.98 + epsilon) * h * (deep - y)) + 8
        x_max = x[-1] - 8
        x_true = np.sqrt(2 * (0.98) * h * (deep - 0.5))
        plt.plot(x, y, color='lightskyblue', linewidth=2)

        ##ground
        x = np.linspace(0, 20, num=50)
        y = np.zeros(50) + 0.2
        plt.plot(x, y, color='black', linewidth=2)

        ax.set_xlim((0, 20))
        ax.set_ylim((0, 20))

        plt.axis('off')
        i = i + 1
        if count == 4:
            plt.savefig('./causal_data/flow/{}_{}_{}/test/a_'.format(args.log_dir, args.ratio, args.epochs) + str(int(data[0])) + "_" + str(int(data[1])) + "_" + str(int(data[2])) + "_" + str(int(data[3])) + "_" + str(i) + '.png', dpi=96)
            count = 0
        else:
            plt.savefig('./causal_data/flow/{}_{}_{}/train/a_'.format(args.log_dir, args.ratio, args.epochs) + str(int(data[0])) + "_" + str(int(data[1])) + "_" + str(int(data[2])) + "_" + str(int(data[3])) + "_" + str(i) + '.png', dpi=96)
            count += 1

        plt.clf()
