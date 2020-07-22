__author__ = 'Steffen'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import euclidean

def plot_training(filename, stats, labels):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    x = range(len(stats[0]))
    for idx, stat in enumerate(stats):
        if idx == len(stats) - 1:
            ax2 = ax1.twinx()
            ax2.plot(x, stat, label=labels[idx], color='r')
            ax2.set_ylabel('Posterior loss term')
        else:
            ax1.plot(x, stat, label=labels[idx])
            ax1.set_ylabel('Reconstruction and total loss')
    fig.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()

def plot_to_file(filename, waveform, reconstr):

    x = range(len(waveform))
    plt.plot(x, waveform, 'b')
    plt.plot(x, reconstr, 'r')
    plt.plot(x, np.abs(reconstr - waveform), 'k.')
    plt.savefig(filename)
    plt.close()

def generate_sprite_img(filename, samples):
    rows = int(np.ceil(np.sqrt(samples.shape[0])))
    sprite_img = np.zeros((60*rows, 60*rows, 3))
    for idx, s in enumerate(samples):
        plt.figure(figsize=(.6,.6), dpi=10)
        x = range(len(s))
        plt.plot(x, s, 'b')

        buffer = BytesIO()
        plt.savefig(buffer, format='bmp')
        plt.close()

        buffer.seek(0)
        img = Image.open(buffer)
        img = img.convert("RGB")

        buffer.close()

        start_col = (idx % rows) * 60
        start_row = int(idx / rows) * 60
        sprite_img[start_row:start_row+60, start_col:start_col+60, :] = np.array(img).reshape(60,60,3)

    plt.imsave(filename, sprite_img)

def plot_distance(latents1, latents2, filename='path.png'):
    dist = [euclidean(l1, l2) for l1, l2 in zip(latents1, latents2)]
    plot_data_single(dist, filename)

def plot_data(x, y, filename):
    plt.plot(x, y)

    plt.savefig(filename)
    plt.close()

def plot_data_single(y, filename):
    plot_data(range(len(y)), y, filename)