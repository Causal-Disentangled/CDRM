import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data, plot_file, keys=None, 
                clip=True, label_min=True, label_end=True):
    if not keys:
        keys = data.keys()
    plt.figure()
    for i,key in enumerate(keys):
        plt.subplot(len(keys),1,i+1)
        if clip:
            limit = 2*np.mean(np.abs(data[key]))
            y = np.clip(data[key],-limit,limit)
        else:
            y = data[key]
        plt.plot(y, linewidth=1.,label=key)
        if label_min:
            plt.plot(np.argmin(data[key]),np.min(data[key]),'o',
                    label="min: {:.3g}".format(np.min(data[key])))
        if label_end:
            plt.plot(len(data[key])-1,data[key][-1],'o',
                    label="end: {:.3g}".format(data[key][-1]))
        plt.legend()
    plt.savefig(plot_file)
    plt.close()