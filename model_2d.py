__author__ = 'Steffen'

import tensorflow as tf
import numpy as np
import math

# 2-D convolutions
# Assumes #channelsx#samples
nchannel = 19

def inv_eegnet_decoder(input, n_hidden, n_output, keep_prob):
    input_size = 64

    #  reshape to NHWC
    x = tf.reshape(input, [-1, 1, 1, input_size])

    # expand to pre-pool size from encoder
    x = tf.tile(x, [1, global_pool_size[0], global_pool_size[1], 1])

    # now do the convolutions
    with tf.variable_scope('eegnet_decoder'):

        # block 4
        x = tf.layers.conv2d_transpose(x, n_hidden * 8, kernel_size=(1,10), activation=tf.nn.elu)
        x = x[:output_shapes[3][0], :output_shapes[3][1], :output_shapes[3][2], :output_shapes[3][3]]

        # block 3
        x = tf.layers.conv2d_transpose(x, n_hidden * 4, kernel_size=(1,10), activation=tf.nn.elu)
        x = tf.image.resize_images(x, [x.get_shape()[1].value, x.get_shape()[2].value * 3])
        x = x[:output_shapes[2][0], :output_shapes[2][1], :output_shapes[2][2], :output_shapes[2][3]]

        # block 2
        x = tf.layers.conv2d_transpose(x, n_hidden * 2, kernel_size=(1,10), activation=tf.nn.elu)
        x = tf.image.resize_images(x, [x.get_shape()[1].value, x.get_shape()[2].value * 3])
        x = x[:output_shapes[1][0], :output_shapes[1][1], :output_shapes[1][2], :output_shapes[1][3]]

        # block 1
        x = tf.layers.conv2d_transpose(x, n_hidden, kernel_size=(nchannel, 1), activation=tf.nn.elu)
        x = tf.layers.conv2d_transpose(x, n_hidden, kernel_size=(1, 3), activation=tf.nn.elu)
        x = x[:output_shapes[0][0], :output_shapes[0][1], :output_shapes[0][2], :output_shapes[0][3]]

        x = tf.layers.conv2d(x, 1, kernel_size=(1,1))
        x = tf.layers.flatten(x)

    return x

def eegnet_encoder(input, n_hidden, n_output, keep_prob):
    with tf.variable_scope('eegnet_encoder'):
        # channel x samples
        x = tf.reshape(input, [-1, nchannel, 128, 1])

        global output_shapes
        output_shapes = []
        output_shapes.append(x.get_shape())

        # block 1
        x = tf.layers.conv2d(x, n_hidden, kernel_size=(1, 3), activation=tf.nn.elu, padding='same')
        x = tf.layers.conv2d(x, n_hidden, kernel_size=(nchannel, 1), activation=tf.nn.elu)
        # maxpool - not necessary?
        output_shapes.append(x.get_shape())

        # block 2
        x = tf.layers.conv2d(x, n_hidden * 2, kernel_size=(1,10), activation=tf.nn.elu, padding='same')
        x = tf.layers.max_pooling2d(x, pool_size=(1,3), strides=(1,3))
        output_shapes.append(x.get_shape())

        # block 3
        x = tf.layers.conv2d(x, n_hidden * 4, kernel_size=(1,10), activation=tf.nn.elu, padding='same')
        x = tf.layers.max_pooling2d(x, pool_size=(1,3), strides=(1,3))
        output_shapes.append(x.get_shape())
        # maxpool

        # block 4
        x = tf.layers.conv2d(x, n_hidden * 8, kernel_size=(1,10), activation=tf.nn.elu, padding='same')

        global global_pool_size
        global_pool_size= (x.get_shape()[1].value, x.get_shape()[2].value)

        x = tf.layers.conv2d(x, n_output * 2,
                         kernel_size=(1, 1),
                         activation=None)

        x = tf.layers.average_pooling2d(x, pool_size=global_pool_size, strides=(1, 1))

        # output layer
        gaussian_params = tf.layers.flatten(x)

        mean = gaussian_params[:, :n_output]
        stddev = tf.nn.softplus(gaussian_params[:, n_output:] + 1e-8)

    return mean, stddev


def eegnet_decoder(input, n_hidden, n_output, keep_prob):
    with tf.variable_scope('eegnet_decoder'):

        new_dim = global_pool_size[1]
        out_samples = tf.cast(n_output / nchannel, tf.int32)


        size = np.sqrt(input.get_shape()[1].value).astype(int)
        x = tf.reshape(input, [-1, size, size, 1])

        num_conv = np.ceil(math.log(n_output / input.get_shape()[1].value, 4)).astype(int)

        for i in range(num_conv):
            coefficient = np.power(2, num_conv - i - 1)
            n_filters = n_hidden * coefficient
            x = tf.layers.conv2d(x, n_filters, kernel_size=(3,3), activation=tf.nn.elu)
            width = size * np.power(2, i + 1).astype(int)
            x = tf.image.resize_images(x, [width, width])


        # channel x samples
        x = tf.layers.conv2d(x, 1, kernel_size=(1,1), activation=tf.nn.sigmoid)

    return tf.layers.flatten(x)[:, :n_output]


def conv2d_block(input, n_hidden, keep_prob=0.9, filter_size=(2,2), dilation_size=(1,1), pool=False):
    pool_size = 2
    # try dilations later
    conv = tf.layers.conv2d(input, n_hidden, filter_size, dilation_rate=dilation_size,
                            activation=tf.nn.elu, padding='same')
    drop = tf.nn.dropout(tf.nn.elu(conv), keep_prob)
    if pool:
        drop = tf.layers.max_pooling2d(drop, pool_size=(pool_size,pool_size), strides=(pool_size,pool_size))
    return drop

def residual_decoder(input, n_hidden, n_output, keep_prob):

    x = tf.reshape(input, [-1, 8, 8, 1])
    width = np.ceil(np.sqrt(n_output)).astype(int)
    x = tf.image.resize_images(x, [width,width])

    n_res = 100

    for i in range(n_res):
        for i in range(2):
          residual_block(x, n_hidden * 2**(n_res % 6- i - 1), filter_size=(2,2))

    x = tf.layers.conv2d(x, 1, (1,1), activation=tf.nn.sigmoid)

    return tf.layers.flatten(x)[:,:n_output]


def conv2d_decoder(input, n_hidden, n_output, keep_prob):

    n_sample = 128

    dim_z = 64

    init_c_dim = np.ceil(np.sqrt(dim_z / 6)).astype(int)
    pad_len = init_c_dim * 6 * init_c_dim - dim_z

    x = tf.pad(input, [[0, 0], [0, pad_len]])
    x = tf.reshape(x, [-1, init_c_dim, 6 * init_c_dim, 1])

    # conv transpose
    # 3 -> 3x3, stride 1x2, n_hidden for each conv
    # 4 -> 3x3*2**i, stride 1x2, n_hidden*2**(3 - i)

    for i in range(4):
        x = tf.layers.conv2d_transpose(x, n_hidden, kernel_size=(2,2),
                                       activation=tf.nn.elu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.conv2d_transpose(x, n_hidden, kernel_size=(2,2),
                                       activation=tf.nn.elu)
        x = tf.layers.batch_normalization(x)
        new_size = [int(n.value * 2) for n in x.get_shape()[1:3]]
        x = tf.image.resize_images(x, new_size)

    n_layers = 4
    filter_size = 10

    # 3 -> n_hidden, (1,filter_size)
    # 4 -> n_hidden*2**i, (1,filter_size * (n_layers - i))
    # 4r -> n_hidden*2**(n_layers-i-1), (1,filter_size*(1+i))

    #x = tf.reshape(x, [-1, nchannel,n_sample, x.get_shape()[3].value])
    x = tf.image.resize_images(x, [nchannel, n_sample])

    for i in range(n_layers):
        x = tf.layers.conv2d(x, n_hidden, kernel_size=(1,filter_size),
                             activation=tf.nn.elu, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.layers.conv2d(x, n_hidden, kernel_size=(2,2), activation=tf.nn.elu, padding='same')

    #x = tf.image.resize_images(x, [width,width])

    x = tf.layers.conv2d(x, 1, (1,1), activation=tf.nn.sigmoid)

    return tf.layers.flatten(x)

def residual_block(input, num_filters, strides=(1,1), filter_size=(3,3)):

    out = tf.layers.batch_normalization(input)
    out = tf.nn.elu(out)
    out = tf.layers.conv2d(out, filters=num_filters, kernel_size=filter_size, strides=strides, padding='same')
    out = tf.layers.batch_normalization(out)

    out = tf.nn.elu(out)
    out = tf.layers.conv2d(out, filters=num_filters, kernel_size=filter_size, padding='same')

    return out + input


def conv_upsample_block(input, n_hidden, keep_prob):
    conv = tf.layers.conv2d_transpose(input, filters=n_hidden, activation=tf.nn.elu, kernel_size=(2,2),
                                      strides=(1,2), padding='same')
    drop = tf.nn.dropout(conv, keep_prob)
    out = tf.layers.batch_normalization(drop)
    return out

def convolutional_encoder(x, n_hidden, n_output, keep_prob):

    with tf.variable_scope('conv_enc'):
        x = tf.reshape(x, [-1, nchannel, 128, 1]) # un-hardcode this later

        #out = tf.layers.conv2d(x, n_hidden, (1,4), activation=tf.nn.elu, padding='same')

        #out = tf.layers.conv2d(out, n_hidden, (nchannel,1), activation=tf.nn.elu, padding='same')

        out = conv2d_block(x, n_hidden)
        
        for j in range(5):
            new_dim = n_hidden * 2 ** int(j/3)
            out = tf.layers.conv2d(out, new_dim, (1,1), padding='same')
            for i in range(4):
                out = residual_block(out, new_dim)


        out = tf.layers.dense(tf.layers.flatten(out), n_output * 2)

        gaussian_params = tf.layers.flatten(out)

        #gaussian_params = tf.layers.dense(out, n_output * 2)

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = tf.nn.softplus(gaussian_params[:, n_output:] + 1e-8)

    return mean, stddev

def convolutional_decoder(x, n_hidden, n_output, keep_prob):

    with tf.variable_scope('conv_dec'):

        # Linear layer
        out = tf.layers.dense(x, n_hidden * n_hidden, activation=tf.nn.elu)
        out = tf.reshape(out, [-1, n_hidden, n_hidden, 1])

        for j in reversed(range(5)):
            new_dim = n_hidden * 2 ** int(j/3)
            out = tf.layers.conv2d(out, new_dim, (1,1), padding='same')
            for i in range(4):
                out = residual_block(out, new_dim)

        # Final convolution for output
        conv3 = tf.layers.conv2d(out, filters=1, activation=tf.nn.sigmoid, kernel_size=(1,1))
        output = tf.layers.dense(tf.layers.flatten(conv3), n_output)

    return output

def ladder_encoder(x, n_hidden, n_output, keep_prob=1.0, n_layers=5):

    output_mu = tf.Variable(tf.zeros([128,0]))
    output_sd = tf.Variable(tf.zeros([128,0]))

    latent_per_layer = int(n_output / n_layers)

    with tf.variable_scope('ladder_enc'):
        x = tf.reshape(x, [-1, nchannel, 128, 1]) # un-hardcode this later

        out = tf.layers.conv2d(x, n_hidden, (1,2), activation=tf.nn.elu, padding='same')
        out = tf.layers.conv2d(out, n_hidden, (nchannel,1), activation=tf.nn.elu)

        for layer in range(n_layers):
            out = residual_block(out, n_hidden)
            z_out = tf.layers.dense(tf.layers.flatten(out), latent_per_layer * 2)
            output_mu = tf.concat([output_mu, z_out[:,:latent_per_layer]], axis=1)
            output_sd = tf.concat([output_sd, z_out[:,latent_per_layer:]], axis=1)

    return output_mu, tf.nn.softplus(output_sd)

def ladder_linear_encoder(x, n_hidden, n_output, keep_prob=1.0, n_layers=5):

    output_mu = tf.Variable(tf.zeros([128,0]))
    output_sd = tf.Variable(tf.zeros([128,0]))

    latent_per_layer = int(n_output / n_layers)

    out = x
    with tf.variable_scope('ladder_enc'):
        for layer in range(n_layers):
            out = tf.layers.dense(out, n_hidden, activation=tf.nn.elu)
            latent_out = tf.layers.dense(out, latent_per_layer * 2)
            output_mu = tf.concat([output_mu, latent_out[:,:latent_per_layer]], axis=1)
            output_sd = tf.concat([output_sd, latent_out[:,latent_per_layer:]], axis=1)
    return output_mu, tf.nn.softplus(output_sd)


def ladder_res_decoder(x, n_hidden, dim_z, n_output, n_layers=5):

    latent_per_layer = int(dim_z / n_layers)
    hidden = tf.Variable(tf.zeros([128,0]))
    x = tf.reverse(x, axis=[1])

    with tf.variable_scope('ladder_dec'):

        for layer in range(n_layers):
            start_idx = layer * latent_per_layer
            latent = x[:,start_idx:start_idx + latent_per_layer]
            hidden = tf.concat([hidden, latent], axis=1)
            hidden = tf.reshape(hidden, [-1, 1, hidden.get_shape()[1], 1])
            hidden = tf.layers.conv2d(hidden, n_hidden, (1,1), padding='same')
            hidden = residual_block(hidden, n_hidden)
            hidden = tf.layers.flatten(hidden)

    final_out = tf.layers.dense(hidden, n_output, activation=tf.nn.sigmoid)
    return final_out

def ladder_sum_encoder(x, n_hidden, n_output, n_layers=5):

    output_mu = tf.Variable(tf.zeros([128,n_output]))
    output_sd = tf.Variable(tf.zeros([128,n_output]))

    out = x
    with tf.variable_scope('ladder_enc'):
        for layer in range(n_layers):
            out = tf.layers.dense(out, n_hidden, activation=tf.nn.elu)
            z_out = tf.layers.dense(out, n_output * 2)
            output_mu += z_out[:,:n_output]
            output_sd += z_out[:,n_output:]
    return output_mu, tf.nn.softplus(output_sd)

def ladder_sum_decoder(x, n_hidden, dim_z, n_output, n_layers=5):

    hidden = tf.Variable(tf.zeros([128,0]))
    out = tf.Variable(tf.zeros([128,n_output]))

    with tf.variable_scope('ladder_dec'):

        for layer in range(n_layers):
            hidden = tf.concat([hidden, x], axis=1)
            hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.elu)
            out += tf.layers.dense(hidden, n_output)
            out = tf.nn.sigmoid(out)

    return out


def ladder_decoder(x, n_hidden, dim_z, n_output, keep_prob=1.0, n_layers=5):

    latent_per_layer = int(dim_z / n_layers)
    hidden = tf.Variable(tf.zeros([128,0]))
    x = tf.reverse(x, axis=[1])

    with tf.variable_scope('ladder_dec'):

        for layer in range(n_layers):
            start_idx = layer * latent_per_layer
            latent = x[:,start_idx:start_idx + latent_per_layer]
            hidden = tf.concat([hidden, latent], axis=1)
            hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.elu)

    final_out = tf.layers.dense(hidden, n_output, activation=tf.nn.sigmoid)
    return final_out

def conv_decoder():
    tf.layers.conv2d_transpose()