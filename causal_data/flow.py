import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.image as mpimg
import random
import math
import numpy as np

if not os.path.exists('causal_data/original_flow/'):
    os.makedirs('causal_data/original_flow/train/')
    os.makedirs('causal_data/original_flow/test/')

if not os.path.exists('causal_data/control_flow/'):
    os.makedirs('causal_data/control_flow/train/')
    os.makedirs('causal_data/control_flow/test/')

count = 0
i = 0
flow_dir = 'causal_data/original_flow/'
fname = 'flow'
path = os.path.join(flow_dir, fname + '.txt')

for r in range(5, 35):
    for h_raw in range(10, 40):
        for hole in range(6, 15):
            ball_r = r/30.0
            h = pow(ball_r,3)+h_raw/10.0
            deep = hole/3.0
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            ax = plt.gca()
    
            # water in cup 
            rect = plt.Rectangle(([3, 0]),5,5+h,color='lightskyblue')
            ax.add_artist(rect)
            ball = plt.Circle((5.5,+ball_r+0.5), ball_r, color = 'firebrick')

            ## cup
            left = plt.Polygon(([3, 0],[3, 19]), color = 'black', linewidth = 2)
            right_1 = plt.Polygon(([8, 0],[8, deep]), color = 'black', linewidth = 2)
            right_2 = plt.Polygon(([8, deep+0.4],[8, 19]), color = 'black', linewidth = 2)
            ax.add_artist(left)
            ax.add_artist(right_1)
            ax.add_artist(right_2)
            ax.add_artist(ball)
    
            #water line
            y = np.linspace(deep,0.5)
            epsilon = 0.01 * np.max([np.abs(np.random.randn(1)),1])
            x = np.sqrt(2*(0.98+epsilon)*h*(deep-y))+8
            x_max = x[-1]-8
            x_true = np.sqrt(2*(0.98)*h*(deep-0.5))
            plt.plot(x,y,color='lightskyblue',linewidth = 2)
    
            ##ground
            x = np.linspace(0,20,num=50)
            y = np.zeros(50)+0.2
            plt.plot(x,y,color='black',linewidth = 2)
            
            ax.set_xlim((0, 20))
            ax.set_ylim((0, 20))
    
            plt.axis('off')

            a = (random.randint(0, 3))
            i = i + 1
            data = [str(int(r)), str(int(h)), str(int(x_true * 10)), str(int(hole))]#Our data
            del data[a]
            control_data = [str(int(r)), str(int(h)), str(int(x_true * 10)), str(int(hole))]#CausalVAE data
            if a == 0:
                control_data[a] = str(int(19))
            if a == 1:
                control_data[a] = str(int(2))
            if a == 2:
                control_data[a] = str(int(38))
            if a == 3:
                control_data[a] = str(int(10))


            if count == 4:
                plt.savefig('./causal_data/original_flow/test/a_' + data[0] + "_" + data[1] + "_" + data[2] + "_" + str(i) + '.png', dpi=96)
                count = 0
            else:
                plt.savefig('./causal_data/original_flow/train/a_' + data[0] + "_" + data[1] + "_" + data[2] + "_" + str(i) + '.png', dpi=96)
                count += 1

            if count == 4:
                plt.savefig('./causal_data/control_flow/test/a_' + control_data[0] + "_" + control_data[1] + "_" + control_data[2] + "_" + control_data[3] + "_" + str(i) + '.png', dpi=96)
                count = 0
            else:
                plt.savefig('./causal_data/control_flow/train/a_' + control_data[0] + "_" + control_data[1] + "_" + control_data[2] + "_" + control_data[3] + "_" + str(i) + '.png', dpi=96)
                count += 1

            with open(path, 'a') as f:
                f.write(str(int(r)) + " " + str(int(h)) + " " + str(int(x_true * 10)) + " " + str(int(hole)) + " " + str(int(i)) + " " + str(int(a)) + '\n')

            plt.clf()
