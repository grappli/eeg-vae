#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from vae_architecture import EEGNetVAELayers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.slim.nets import resnet_v2

print(tf.__version__)
import tensorflow.random

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import PIL
import imageio
import eeg_data
import math

from IPython import display

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


# In[2]:


def resnet_12(inputs, num_classes, scope='resnet_12'):
    blocks = [resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=2, stride=1),
              resnet_v2.resnet_v2_block('block2', base_depth=64, num_units=2, stride=1),
              resnet_v2.resnet_v2_block('block3', base_depth=64, num_units=2, stride=1),
              resnet_v2.resnet_v2_block('block4', base_depth=64, num_units=2, stride=1),
              resnet_v2.resnet_v2_block('block5', base_depth=64, num_units=2, stride=1)]
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=True,
                   global_pool=True, output_stride=None,
                   include_root_block=True,
                   reuse=None, scope=scope)


# In[3]:


def get_overlap(data, percent_overlap, seq_len=128):
    stride_size = seq_len - int(percent_overlap * seq_len)
    num_seq = int((len(data) - seq_len) / stride_size) + 1
    for i in range(num_seq):
        yield data[i*stride_size:i*stride_size+seq_len]

train_total_raw, train_total_data, train_overlap_data, test_raw, test_data, test_overlap_data, _, _, segment_indices = eeg_data.get_eeg_data()

#train_total_data = np.reshape(train_overlap_data,
#            (train_overlap_data.shape[0], train_overlap_data.shape[1], train_overlap_data.shape[2], 1)).astype(np.float32)
#test_data = np.reshape(test_overlap_data, 
#            (test_overlap_data.shape[0], test_overlap_data.shape[1], test_overlap_data.shape[2], 1)).astype(np.float32)

percent_overlap = 0.0


train_data = np.moveaxis(train_total_raw, -1, 1).reshape((-1,19))
test_data = np.moveaxis(test_raw, -1, 1).reshape((-1,19))

train_total_data = np.expand_dims(np.moveaxis(list(get_overlap(train_data, percent_overlap)), 1, -1), -1).astype(np.float32)
test_data = np.expand_dims(np.moveaxis(list(get_overlap(test_data, percent_overlap)),1, -1), -1).astype(np.float32)

# normalize data
train_total_data -= train_total_data.min(axis=2,keepdims=True)
train_total_data /= train_total_data.max(axis=2,keepdims=True)
test_data -= test_data.min(axis=2,keepdims=True)
test_data /= test_data.max(axis=2,keepdims=True)

n_channel = 19 # 19 channels
n_sample = 128
train_total_data = train_total_data[:,list(range(n_channel)),:,:]
test_data = test_data[:,list(range(n_channel)),:,:]

#train_total_data = train_total_data.reshape((-1,1,n_sample,1))
#test_data = test_data.reshape((-1,1,n_sample,1))

print(train_total_data.shape)
print(test_data.shape)


# In[4]:


TRAIN_BUF = 60000
BATCH_SIZE = 500

class GraphDataset:
    def __init__(self, data):
        self.data = data
    
    def from_data(data):
        d = data.copy()
        ds = GraphDataset(d)
        return ds
    
    def shuffle(self, buff=0):
        np.random.shuffle(self.data)
        return self
        
    def batch(self, size):
        self.batch_size = 100
        return self
        
    def batches(self):
        total_batches = np.int64(np.ceil(self.data.shape[0] / self.batch_size))
        for i in range(total_batches):
            start = i*self.batch_size
            end = min((i+1)*self.batch_size, self.data.shape[0])
            yield self.data[start:end]

TEST_BUF = 10000
print(train_total_data.shape)
train_dataset = GraphDataset.from_data(train_total_data).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = GraphDataset.from_data(test_data).shuffle(TEST_BUF).batch(BATCH_SIZE)



# In[ ]:


tensorboard_logdir = "/cache/tensorboard-logdir/vanilla_vae"
global_step = 0
writer = tf.contrib.summary.create_file_writer(tensorboard_logdir)


# In[ ]:


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, hidden_dim, architecture="mlp", loss_type='vanilla', beta=1.0):
        super(CVAE, self).__init__()

        self.session = tf.Session()
        
        self.beta = beta
        self.loss_type = loss_type
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        pad_amt = 12 - (latent_dim % 12) 
        channel_dim = int(np.ceil(latent_dim / 12))

        conv_shape = (1,12,channel_dim)

        if architecture == 'mlp':
            input_shape = (n_channel,n_sample,1)
            self.inference_net = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim*3, name='z_out')   
            ])

            self.generative_net = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(input_dim)
            ])
        elif architecture == 'conv2d':
            input_shape = (n_channel,n_sample)
            self.inference_net = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv1D(hidden_dim, kernel_size=8, strides=4, activation=tf.nn.relu),
                tf.keras.layers.Reshape((n_sample,hidden_dim)),
                tf.keras.layers.Conv1D(hidden_dim, kernel_size=8, strides=4, activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(hidden_dim, kernel_size=3, activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.layers.Flatten(),
                tf.layers.Dense(latent_dim*3, name='z_out')
            ])
            self.generative_net = tf.keras.Sequential([
                tf.keras.layers.Reshape((self.latent_dim,1)),
                tf.keras.layers.ZeroPadding1D((0,pad_amt)),
                tf.keras.layers.Reshape(conv_shape),
                tf.keras.layers.Conv2DTranspose(hidden_dim, kernel_size=(1,3), activation=tf.nn.relu),
                tf.keras.layers.UpSampling2D(size=(1,3)),
                tf.keras.layers.Conv2DTranspose(hidden_dim, kernel_size=(1,3), activation=tf.nn.relu),
                tf.keras.layers.UpSampling2D(size=(1,3)),
                tf.keras.layers.Cropping2D(((0,0), (2,2))),
                tf.keras.layers.Conv2DTranspose(hidden_dim, kernel_size=(n_channel, 1), activation=tf.nn.relu),
                tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same')
            ])    

        self.inputs = tf.placeholder(shape=[None] + list(input_shape), dtype=tf.float32)

        self.loss = self.compute_loss()
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())
        
        
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def reconstruct(self, x):    
        return self.session.run(self.outputs, feed_dict={self.inputs: x})

    def encode(self):
        mean, logvar, iaf_h = tf.split(self.inference_net(self.inputs), num_or_size_splits=3, axis=1)
        return mean, logvar, iaf_h

    def latent_code(self, x):
        return self.session.run(self.z, feed_dict={self.inputs: x})

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        logits = self.generative_net(z)
        self.outputs = tf.sigmoid(logits)
        return logits

    def iaf(self, mean, logvar, h, n_transforms=10): 
        eps = tf.random.normal(tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        
        logq_zx = -tf.reduce_sum(0.5 * logvar + 0.5 * tf.square(eps) + 0.5 * np.log(2*np.pi), axis=1)

        base_dist = tfd.MultivariateNormalDiag(tf.zeros_like(mean))

        for i in range(n_transforms):
            
            z = tf.reverse(z, axis=[1])
            inp = tf.concat((z,h), axis=1)
            
            m1 = tfb.masked_dense(inp, units=self.hidden_dim, num_blocks=self.latent_dim*2-1, activation=tf.nn.relu, exclusive=True)
            m2 = tfb.masked_dense(m1, units=self.hidden_dim, num_blocks=self.hidden_dim*2-1, activation=tf.nn.relu)
            
            mu = tf.layers.dense(m2, units=self.latent_dim)
            sigma = tf.layers.dense(m2, units=self.latent_dim, activation=tf.nn.sigmoid)            
            
            z = sigma * z + (1 - sigma) * mu
            
            logq_zx -= tf.reduce_sum(tf.log(sigma), axis=1)

        logp_z = base_dist.log_prob(z)
        return  logq_zx - logp_z, z

    def compute_loss(self, which_set='train'):
        mean, logvar, iaf_h = self.encode()
        if self.loss_type == 'mmd':
            self.z = mean
            true_samples = tf.random.normal(shape=tf.shape(mean))
            posterior_loss = compute_mmd(true_samples, self.z)
        elif self.loss_type == 'iaf':
            posterior_loss, self.z = self.iaf(mean, logvar, iaf_h)

        else:
            self.z = self.reparameterize(mean, logvar)

            logpz = log_normal_pdf(self.z, tf.zeros_like(mean), tf.zeros_like(logvar)) 
            logqz_x = log_normal_pdf(self.z, mean, logvar)
            posterior_loss = logqz_x - logpz

        self.posterior_loss = self.beta * tf.reduce_mean(posterior_loss)

        x_logit = tf.reshape(self.decode(self.z), tf.shape(self.inputs))

        self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=self.inputs))
        total_loss = self.recon_loss + self.posterior_loss

        return total_loss



class CVAE1D(CVAE):
    def __init(self, latent_dim, input_dim, hidden_dim, kernel_size=3):
        
        self.latent_dim = latent_dim
        super(CVAE1D, self).__init__(latent_dim, input_dim, hidden_dim, architecture='mlp')
        
        if architecture == 'mlp':
            self.inference_net = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim*3)   
            ])
        else:
            self.inference_net = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Conv1D(hidden_dim, kernel_size=kernel_size, activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(hidden_dim, kernel_size=kernel_size, activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Dense(latent_dim*3)   
            ])
        
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(input_dim)
        ])


# In[ ]:


def plot_reconstruction_simple(input_sequence, reconstruction, latent_dim):
    plt.clf()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    
    x = list(range(len(input_sequence)))
    ax.plot(x, input_sequence)
    ax.plot(x, reconstruction)
    
    ax.set_ylabel("Latent dim = {}".format(latent_dim))
    

def plot_reconstruction(input_sequence, reconstruction, subplot_dims):
    
    x = list(range(len(input_sequence)))
    ax = plt.subplot(3, 4, subplot_dims)
    ax.cla()
    
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    
    ax.plot(x, input_sequence)
    ax.plot(x, reconstruction)
    
    x_idx = subplot_dims % 4
    if x_idx == 1:
        ax.set_ylabel("Latent dim = {}".format(latent_dims[int(subplot_dims/4)]))
    
    if subplot_dims > 8:
        ax.set_xlabel("Hidden dim = {}".format(hidden_dims[int(x_idx-1)]))
    
def display_reconstruction(subplot_dims, mmd_loss=False, iaf=False):
    sample = np.expand_dims(test_data[1], axis=0)
    reconstruction = model.reconstruct(sample)
    sample = sample.reshape((1,1,128,-1))
    reconstruction = reconstruction.reshape((1,1,128,-1))
    plot_reconstruction(sample[0,0,:,0], reconstruction[0,0,:,0], subplot_dims)


# In[ ]:


def log_normal_pdf(sample, mean, logvar):
    base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.maximum(tf.exp(0.5*logvar), 1e-15))
    return base_dist.log_prob(sample)

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    val = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
    del tiled_x, tiled_y, x_size, y_size
    return val

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    del x_kernel, y_kernel, xy_kernel
    return mmd


def train(model, x):
    model.session.run([model.train_op], 
                      feed_dict={model.inputs: x})

def get_loss(model, x):
    return model.session.run([model.loss, 
                              model.posterior_loss, 
                              model.recon_loss], 
                             feed_dict={model.inputs: x})


# In[ ]:



def calculate_distance(model):
    global segment_indices, test_data, train_total_data
    segment_indices = segment_indices.astype(np.int32)
    num_seqs = segment_indices[2] - segment_indices[1]
    num_seqs = 100
    n_random = 1000
    
    test_copy = test_data.copy()
    
    latents = model.latent_code(test_data[:num_seqs])
    seq_latents = np.array(latents)
    
    dist = np.mean(np.sqrt(np.sum(np.square(np.diff(latents,axis=0)), axis=1)))
    seq_dist = dist
    print("Mean sequence distance: {}".format(seq_dist))
    rand_len = 0
    shorter = 0
    for i in range(n_random):

        np.random.shuffle(test_copy)
        latents = model.latent_code(test_copy[:num_seqs])
        dist = np.mean(np.sqrt(np.sum(np.square(np.diff(latents,axis=0)), axis=1)))
        
        rand_len += dist
        if seq_dist < dist:
            shorter += 1
            
            
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(latents)

    X_pca = pca.transform(np.concatenate((latents, seq_latents), axis=0))
    plt.scatter(X_pca[:num_seqs, 0], X_pca[:num_seqs, 1], color='tab:blue')
    plt.scatter(X_pca[num_seqs:, 0], X_pca[num_seqs:, 1], color='tab:orange')
    plt.show()
        
            
    print("Mean random distance: {}".format(rand_len / n_random))

    print("Non-random is shorter {}% of the time".format(100.0 * shorter / n_random))
    print("Non-random is {}% longer than random.".format(100.0 * (seq_dist - (rand_len / n_random)) / (rand_len / n_random)))
    return shorter / n_random


# In[ ]:


def show_example_reconstruction(model):
    print("Loss={}".format(model.loss_type))
    sample = np.expand_dims(test_data[1], axis=0)
    reconstruction = model.reconstruct(sample)
    sample = sample.reshape((1,1,n_sample,-1))
    reconstruction = reconstruction.reshape((1,1,n_sample,-1))
    plot_reconstruction_simple(sample[0,0,:,0], reconstruction[0,0,:,0], latent)

    plt.show()


# In[ ]:


epochs = 50
latent_dims = [64]
hidden_dims = [512]

input_dim = np.prod(train_total_data.shape[1:])
epoch_losses = np.zeros((3,epochs,3))
logdets = np.zeros((3,epochs))

model_params = [('vanilla', 1e-3), ('iaf', 0), ('mmd', 1)]

for idx_latent, latent in enumerate(latent_dims):
    for idx_hidden, hidden in enumerate(hidden_dims):
        print("latent={}, hidden={}".format(latent, hidden))
        models = [CVAE(latent_dim=latent, 
                      input_dim=input_dim, 
                      hidden_dim=hidden,
                      architecture="conv2d", 
                      loss_type=l,
                      beta=b)
                  for l, b in model_params]                  
                  
        for epoch in range(1, epochs + 1):
            for m_idx, m in enumerate(models):
                start_time = time.time()
                print("Training model, loss={}".format(m.loss_type))
                for train_x in train_dataset.batches():
                    train(m, train_x)

                losses = np.zeros(3)
                num_batches = 0
                
                for test_x in test_dataset.batches():
                    losses += get_loss(m, test_x)
                    num_batches += 1
                    
                avg_losses = losses / num_batches
                epoch_losses[m_idx,epoch - 1] = avg_losses
                                
                zs = m.latent_code(test_data[:100])
                mean_cov = np.mean(np.square(np.cov(zs, rowvar=False) - np.identity(latent)))
                print(mean_cov)
                sign, logdet = np.linalg.slogdet(np.cov(zs, rowvar=False))
                det = sign * np.exp(logdet)
                logdets[m_idx,epoch-1] = det
                print(det)

                end_time = time.time()
                print('Epoch: {}, Test set ELBO: {}, '
                  'time elapsed for current epoch {}'.format(epoch,
                                                        avg_losses[0],
                                                        end_time - start_time))
                show_example_reconstruction(m)
                            
        #display_reconstruction(idx_hidden+(4*idx_latent)+1, mmd_loss=mmd_loss, iaf=use_iaf)


# In[ ]:


import matplotlib.font_manager
plt.clf()
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

from matplotlib import rcParams
from matplotlib import rc
params = {'text.usetex': False, 
          'mathtext.fontset': 'cm', 
          'font.serif':'Computer Modern Roman',
          'font.family': 'serif'}

matplotlib.rcParams.update(params)
for m_idx, m in enumerate(models):
    for p in range(3):
        axes[p].plot(list(range(epochs)), epoch_losses[m_idx,:,p], label=m.loss_type)

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')
axes[0].set_title('Loss total', fontsize=20, y=1.08)
axes[1].set_title('$D_{KL}(q_{\\phi}(z|x)||p(z))$', fontsize=20, y=1.08)
axes[2].set_title('Cross entropy', fontsize=20, y=1.08)

axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')

plt.show()


# In[ ]:


for m_idx, m in enumerate(models):
    plt.plot(list(range(epochs)), logdets[m_idx], label=m.loss_type)
    plt.yscale('log')
plt.legend()
plt.show()


# In[ ]:


for m in models:
    calculate_distance(m)


# In[ ]:


for m in models:
    show_example_reconstruction(m)


# In[ ]:


from scipy.stats import kde
def plot_densities(model):
    plt.clf()
    p_z = np.random.multivariate_normal(np.zeros(2), np.identity(2), size=test_data.shape[0])
    #rand_idxs = np.random.choice(list(range(len(test_data))), size=10000)
    pq_z_x = model.latent_code(test_data)
    pq_z_x = np.array(pq_z_x[:,0:2])
        
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))    
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        
    nbins=300
    
    x = p_z[:,0]
    y = p_z[:,1]
    k = kde.gaussian_kde((x,y))
    xi, yi = np.mgrid[-3:3:nbins*1j, -3:3:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues)
    axes[0].set_title('$p(z)$', fontsize=20, y=1.08)
    axes[0].set_xlim(-3,3)
    axes[0].set_ylim(-3,3)
    
    #plt.ylim(-3,3)
    #plt.xlim(-3,3)
    x = pq_z_x[:,0]
    y = pq_z_x[:,1]
    k = kde.gaussian_kde((x,y))
    xi, yi = np.mgrid[-3:3:nbins*1j, -3:3:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues)
    axes[1].set_title('$q_{\\phi}(z|x)$', fontsize=20, y=1.08)
    #axes[1].set_xlim(-3,3)
    #axes[1].set_ylim(-3,3)

    plt.show()


# In[ ]:


for m in models:
    plot_densities(m)


# In[ ]:





# In[ ]:




