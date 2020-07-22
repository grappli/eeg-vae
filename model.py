import tensorflow as tf
import numpy as np
import model_2d
import sys


nchannel = 19

def convolutional_encoder(x, n_hidden, n_output, keep_prob):

    pool_size = 2

    with tf.variable_scope('conv_enc'):
        x = tf.reshape(x, [-1, x.get_shape()[-1], 1])
        conv1 = tf.layers.conv1d(x, filters=n_hidden, activation=tf.nn.elu, kernel_size=4, dilation_rate=2)
        drop = tf.nn.dropout(conv1, keep_prob)
        mp1 = tf.layers.max_pooling1d(drop, pool_size=pool_size, strides=pool_size)
        conv2 = tf.layers.conv1d(mp1, filters=n_hidden, activation=tf.nn.elu, kernel_size=2, dilation_rate=2)
        drop = tf.nn.dropout(conv2, keep_prob)
        mp2 = tf.layers.flatten(tf.layers.max_pooling1d(drop, pool_size=pool_size, strides=pool_size))

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / model.py
        wo = tf.get_variable('wo', [mp2.get_shape()[1], n_output * 2],
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        bo = tf.get_variable('bo', [n_output * 2], initializer=tf.constant_initializer(0.))
        gaussian_params = tf.matmul(mp2, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = tf.nn.softplus(gaussian_params[:, n_output:] + 1e-8)

    return mean, stddev

def conv_upsample_block(input, n_hidden, keep_prob):
    conv = tf.layers.conv1d(input, filters=n_hidden, activation=tf.nn.elu, kernel_size=3, padding='same')
    drop = tf.nn.dropout(conv, keep_prob)
    drop = tf.reshape(drop, [-1, 1, drop.get_shape()[1], n_hidden])
    # Use resize images to upsample
    us = tf.image.resize_images(drop, size=[1, drop.get_shape()[2] * 2])
    us = tf.reshape(us, [-1, us.get_shape()[2], n_hidden])
    return us

#def conv1d_upsample_block(input, n_hidden, n_output, keep_prob):
    #conv = tf.contrib.nn.conv1d_transpose(input, filter=n_hidden, stride=2, kernel_size=2, padding='same')
    #return conv

def full_rf_block(input, n_hidden, n_output, keep_prob):

    conv = tf.layers.conv1d(input, filters=n_hidden, activation=tf.nn.elu, kernel_size=8, padding='same')
    conv = tf.layers.conv1d(conv, filters=n_hidden, activation=tf.nn.elu, kernel_size=8, padding='same')
    conv = tf.layers.conv1d(conv, filters=n_hidden, activation=tf.nn.elu, kernel_size=8, padding='same')


def conv1d_shuffle_upsample_block(input, n_hidden, n_output, keep_prob, upsample_amount=2, loop_num=0):
    conv = tf.layers.conv1d(input, filters=n_hidden, activation=tf.nn.elu, kernel_size=2**(loop_num+1), padding='same')
    conv = tf.layers.conv1d(conv, filters=n_hidden, activation=tf.nn.elu, kernel_size=2**(loop_num+1), padding='same')
    conv = tf.layers.conv1d(conv, filters=n_hidden, activation=tf.nn.elu, kernel_size=2**(loop_num+1), padding='same')

    # batch x size x n_filters/2
    out = tf.reshape(conv, [-1, conv.get_shape()[1].value * upsample_amount,
                            tf.cast(n_hidden / upsample_amount, tf.int32)])
    return out

def conv1d_shuffle_decoder(input, n_hidden, n_output, keep_prob):
    # start with 64
    n_latent = 64
    shuf = tf.layers.dense(input, n_latent)


    upsample_factor=2

    x = tf.reshape(shuf, [-1, input.get_shape()[1].value, 1])
    # get number of doubling layers required
    import math
    num_layers = np.ceil(math.log(n_output / n_latent, upsample_factor)).astype(int)
    # need to upscale by 38
    # 2x2x2x2x2x2
    for i in range(num_layers):
        x = conv1d_shuffle_upsample_block(x, n_hidden, n_output, keep_prob, upsample_amount=upsample_factor, loop_num=i)

    # now 64x64 -> trim from edges to arrive at n_output
    #x = tf.layers.conv1d(x, filters=1, kernel_size=1)
    x = tf.layers.flatten(tf.layers.conv1d(x, filters=1, kernel_size=1))
    start = tf.cast((x.get_shape()[1].value - n_output) / 2, tf.int32)
    return x[:, start:start+n_output]

def convolutional_decoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope('conv_dec'):

        # Linear layer
        #x = tf.layers.dense(x, n_output / 8, activation=tf.nn.elu)
        x = tf.reshape(x, [-1, 64, 1])

        act1 = conv_upsample_block(x, n_hidden * 2, keep_prob)
        #act2 = conv_upsample_block(act1, n_hidden * 2, keep_prob)
        #act3 = conv_upsample_block(act2, n_hidden, keep_prob)

        # Final convolution for output
        conv3 = tf.layers.conv1d(act1, filters=1, activation=tf.nn.sigmoid, kernel_size=1)
        output = tf.layers.flatten(conv3)

    return output

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer

        #embeddings = tf.get_variable("embeddings", [256, 16])
        #embedded = tf.nn.embedding_lookup(embeddings, x)
        #embedded = tf.reshape(embedded, [-1, x.get_shape()[-1]*16])

        w0 = tf.Variable(w_init([x.get_shape()[1].value, n_hidden]), 'w0')
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        #h2 = tf.reshape(h2, [-1, n_hidden, 256])
        #h2 = tf.reduce_sum(h2, axis=2)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / model.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = tf.nn.softplus(gaussian_params[:, n_output:] + 1e-8)

    return mean, stddev

def bi_rnn_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("bi_rnn_encoder"):

        x = tf.reshape(x, [-1, nchannel, 128])
        x = tf.transpose(x, perm=[0, 2, 1])

        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
        lstmcell = tf.contrib.rnn.InputProjectionWrapper(cell, x.get_shape()[-1].value, activation=tf.nn.elu)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                    cell_bw=cell,
                                                    dtype=tf.float32,
                                                    inputs=x)

        states = tf.reduce_sum(states, [0,1])
        states = tf.reshape(states, [-1,n_hidden])

        gaussian_params = tf.layers.dense(states, n_output*2)
        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = tf.nn.softplus(gaussian_params[:, n_output:] + 1e-8)

    return mean, stddev

def bi_rnn_decoder(z, n_hidden, n_output, keep_prob):
    init_vector = tf.layers.dense(z, n_hidden)
    #init_vector = tf.reshape(tf.tile(init_vector, [1,2]), [2,-1,n_hidden])
    initial = tf.contrib.rnn.LSTMStateTuple(init_vector, init_vector)
    n_out = tf.cast(n_output / nchannel, tf.int32)
    with tf.variable_scope("bi_rnn_decoder"):
        #x = tf.tile(z, [1, n_out])
        #x = tf.reshape(x, [-1, n_out, z.get_shape()[1].value])
        x = tf.zeros([128, n_out, n_hidden])
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True, activation=tf.nn.elu)
        lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, nchannel)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                     cell_bw=lstm_cell,
                                                     dtype=tf.float32,
                                                     initial_state_fw=initial,
                                                     initial_state_bw=initial,
                                                     inputs=x)
        y = tf.reduce_sum(outputs, 0)
        y = tf.transpose(y, [0, 2, 1])
        y = tf.layers.flatten(y)
        y = tf.nn.sigmoid(y)
    return y


# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)
        #y = tf.reshape(y, [-1, n_output, 256])

    return y
def one_layer_encoder(input, n_hidden, output_size, keep_prob):
    out = tf.layers.dense(input, output_size*2, activation=tf.nn.elu)
    return out[:,:output_size], tf.nn.relu(out[:,output_size:])

def one_layer_decoder(inputs, n_hidden, output_size, keep_prob):
    return tf.layers.dense(inputs, output_size, activation=tf.nn.sigmoid)

def sample(mean, logvar):
    noise = tf.random_normal(tf.shape(mean))
    sample = mean + tf.exp(0.5 * logvar) * noise
    return sample

def iaf(sample, mean, logvar):
    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar))

def KL_iaf(mu, sigma, dim_z):
    z = sample(mu, sigma)

    logqs = iaf(z, mu, sigma)
    L = tf.get_variable("inverse_cholesky", [dim_z, dim_z], dtype=tf.float32, initializer=tf.zeros_initializer)
    diag_one = tf.ones([dim_z], dtype=tf.float32)
    L = tf.matrix_set_diag(L, diag_one)
    mask = np.tril(np.ones([dim_z, dim_z]))
    L = L * mask
    latent_vector = tf.matmul(z, L)
    logps = iaf(latent_vector, tf.zeros_like(mu), tf.zeros_like(sigma))


    KL_divergence = logqs - logps

    return (z, KL_divergence)

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def mmd(z_enc, z_sample):
    x_kernel = compute_kernel(z_sample, z_sample)
    y_kernel = compute_kernel(z_enc, z_enc)
    xy_kernel = compute_kernel(z_sample, z_enc)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# interface for encoder:
# x_hat, n_hidden, dim_z, keep_prob
# interface for decoder:
# z, n_hidden, n_output, keep_prob

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, use_iaf=False, use_mmd=True,
                encoder='convolutional_encoder', decoder='convolutional_decoder', mse=False):


    with tf.variable_scope('autoencoder'):
        dim_img = x_hat.get_shape()[1].value


        # Dynamically choose encoder and decoder
        this_mod = sys.modules[__name__]
        if encoder.startswith('model_2d'):
            encoder_func = getattr(model_2d, encoder.replace('model_2d.', ''))
        else:
            encoder_func = getattr(this_mod, encoder.replace('model_2d.', ''))
        if decoder.startswith('model_2d'):
            decoder_func = getattr(model_2d, decoder.replace('model_2d.', ''))
        else:
            decoder_func = getattr(this_mod, decoder.replace('model_2d.', ''))

        # batch normalize
        # x_hat = tf.layers.batch_normalization(x_hat)

        n_layers=5

        # encoding
        mu, sigma = encoder_func(x_hat, n_hidden, dim_z, keep_prob)

        if use_iaf:
            z, KL_divergence = KL_iaf(mu, sigma, dim_z)
        else:
            # sampling by re-parameterization technique
            z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            if use_mmd:
                KL_divergence = mmd(z, tf.random_normal(tf.stack([200, dim_z])))
            else:
                KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)

        y = decoder_func(z, n_hidden, dim_img, keep_prob=keep_prob)

        # loss
        if mse:
            marginal_likelihood = tf.losses.mean_squared_error(x, y)
        else:
            marginal_likelihood = -tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        marginal_likelihood_grad = tf.gradients(marginal_likelihood, tf.global_variables(scope='autoencoder'))[-1]
        marginal_likelihood_norm = tf.norm(marginal_likelihood_grad, name='norm')

        KL_divergence = tf.reduce_mean(KL_divergence)

        loss = marginal_likelihood + KL_divergence

        return y, z, loss, marginal_likelihood, KL_divergence, marginal_likelihood_norm

def decoder(z, n_hidden, dim_img, keep_prob=1.0):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob, reuse=True)

    return y
