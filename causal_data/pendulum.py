import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd 
if not os.path.exists('./causal_data/original_pendulum/'):
  os.makedirs('./causal_data/original_pendulum/train/')
  os.makedirs('./causal_data/original_pendulum/test/')

if not os.path.exists('causal_data/control_pendulum/'):
    os.makedirs('causal_data/control_pendulum/train/')
    os.makedirs('causal_data/control_pendulum/test/')

count = 0
b = 0
flow_dir = './causal_data/original_pendulum/'
fname = 'pendulum'
path = os.path.join(flow_dir, fname + '.txt')


def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade

scale = np.array([[0,44],[100,40],[7,7.5],[10,10]])

empty = pd.DataFrame(columns=['i', 'j', 'shade','mid'])
for i in range(-40,44):
    for j in range(60,148):
        if j == 100:
            continue
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        theta = i*math.pi/200.0
        phi = j*math.pi/200.0
        x = 10 + 8*math.sin(theta)
        y = 10.5 - 8*math.cos(theta)

        ball = plt.Circle((x,y), 1.5, color = 'firebrick')
        gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

        light = projection(theta, phi, 10, 10.5, 20.5)
        sun = plt.Circle((light,20.5), 3, color = 'orange')

        #calculate the mid index of 
        ball_x = 10+9.5*math.sin(theta)
        ball_y = 10.5-9.5*math. cos(theta)
        mid = (projection(theta, phi, 10.0, 10.5)+projection(theta, phi, ball_x, ball_y))/2 #position
        shade = max(3,abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y))) #lendth

        shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
        
        ax = plt.gca()
        ax.add_artist(gun)
        ax.add_artist(ball)
        ax.add_artist(sun)
        ax.add_artist(shadow)
        ax.set_xlim((0, 20))
        ax.set_ylim((-1, 21))
        new = pd.DataFrame({
                  'i':(i-scale[0][0])/(scale[0][1]-0),
                  'j':(j-scale[1][0])/(scale[1][1]-0),
                  'shade':(shade-scale[2][0])/(scale[2][1]-0),
                  'mid':(mid-scale[2][0])/(scale[2][1]-0)
                  },
                  
                 index=[1])
        empty=empty.append(new,ignore_index=True)
        plt.axis('off')

        a = (random.randint(0, 3))
        data = [str(int(i)), str(int(j)), str(int(shade)), str(int(mid))]
        del data[a]
        b = b + 1
        control_data = [str(int(i)), str(int(j)), str(int(shade)), str(int(mid))]  # CausalVAE data
        if a == 0:
            control_data[a] = str(int(1))
        if a == 1:
            control_data[a] = str(int(93))
        if a == 2:
            control_data[a] = str(int(4))
        if a == 3:
            control_data[a] = str(int(9))
        if count == 4:
            plt.savefig('./causal_data/original_pendulum/test/a_' + data[0] + "_" + data[1] + "_" + data[2] + "_" + str(b) + '.png',dpi=96)
            count = 0
        else:
            plt.savefig('./causal_data/original_pendulum/train/a_' + data[0] + "_" + data[1] + "_" + data[2] + "_" + str(b) + '.png',dpi=96)
            count += 1

        if count == 4:
            plt.savefig('./causal_data/control_pendulum/test/a_' + control_data[0] + "_" + control_data[1] + "_" + control_data[2] + "_" + control_data[3] + "_" + str(b) + '.png',dpi=96)
            count = 0
        else:
            plt.savefig('./causal_data/control_pendulum/train/a_' + control_data[0] + "_" + control_data[1] + "_" + control_data[2] + "_" + control_data[3]+ "_" + str(b) + '.png',dpi=96)
            count += 1

        with open(path, 'a') as f:
            f.write(str(int(i)) + " " + str(int(j)) + " " + str(int(shade)) + " " + str(int(mid)) + " " + str(int(b)) + " " + str(int(a)) + '\n')

        plt.clf()



