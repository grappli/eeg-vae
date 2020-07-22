import tensorflow as tf
import numpy as np
import mnist_data
import eeg_data
import os
import model
import glob
from visualize import plot_to_file, generate_sprite_img, plot_training, plot_distance, plot_data, plot_data_single
import h5py
from tensorflow.contrib.tensorboard.plugins import projector
import scipy



import argparse

SEGMENT_LENGTH = 128 # number of samples per second
nchannel = 19

def save_embedding(input_data, embeddings_array, sess, writer, channel_idx):

    generate_sprite_img('sprite.png'.format(channel_idx), input_data[:100])

    latent_embeddings = tf.Variable(embeddings_array[:100])
    saver = tf.train.Saver([latent_embeddings])

    sess.run(latent_embeddings.initializer)
    config = projector.ProjectorConfig()
    config.model_checkpoint_path = './logs/{}/embedding.ckpt'.format(channel_idx)
    embedding = config.embeddings.add()
    embedding.tensor_name = latent_embeddings.name
    embedding.sprite.image_path = 'sprite.png'.format(channel_idx)
    embedding.sprite.single_image_dim.extend([60, 60])
    projector.visualize_embeddings(writer, config)

    save_path = saver.save(sess, './logs/{}/embedding.ckpt'.format(channel_idx))
    print('Embedding saved to {}'.format(save_path))


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')
    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')
    parser.add_argument('--dim_z', type=int, default='25', help='Dimension of latent vector', required = True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--visualize_sample', type=int, default=-1)
    parser.add_argument('--model_path', type=str, default='./logs/0/model-train_run.ckpt')
    parser.add_argument('--compute_path', action='store_true')
    parser.add_argument('--encoder', type=str, default='model_2d.convolutional_encoder')
    parser.add_argument('--decoder', type=str, default='model_2d.convolutional_decoder')
    parser.add_argument('--iaf', action='store_true')
    parser.add_argument('--mmd', action='store_true')
    parser.add_argument('--mse', action='store_true')
    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main function"""
def main(args, train_data, test_data):

    tf.reset_default_graph()

    """ parameters """

    # network architecture
    n_hidden = args.n_hidden
    dim_input = train_data.shape[1]  # number of samples per segment
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    run_name = "{}_{}_{}_{}".format(args.encoder, ('iaf' if args.iaf else ('mmd' if args.mmd else 'kl')),
                                    str(args.dim_z), args.run_name) + ('_mse' if args.mse else '')
    print(run_name)

    """ prepare data """

    n_train = train_data.shape[0]
    n_test = test_data.shape[0]

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_input], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_input], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    #z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')


    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence, norm =\
        model.autoencoder(x, x, dim_input, dim_z, n_hidden, keep_prob,
                                                                           encoder=args.encoder,
                                                                           decoder=args.decoder,
                                                                           use_iaf=args.iaf,
                                                                           use_mmd=args.mmd,
                                                                           mse=args.mse)

    s1 = tf.summary.scalar('train/loss', loss)
    s2 = tf.summary.scalar('train/neg_marginal_likelihood', neg_marginal_likelihood)
    s3 = tf.summary.scalar('train/KL_divergence', KL_divergence)
    s4 = tf.summary.scalar('norm', norm)
    training_summaries = tf.summary.merge([s1,s2,s3,s4])

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """

    # save
    saver = tf.train.Saver()

    # train
    n_batches_train = np.ceil(n_train / batch_size).astype(int)
    n_batches_test = np.ceil(n_test / batch_size).astype(int)
    min_tot_loss = 1e99

    all_losses = []
    all_likelihoods = []
    all_divergences = []


    with tf.Session() as sess:

        # train model
        if args.visualize_sample == -1:

            # Random shuffling -> disable for now
            # np.random.shuffle(train_data)

            writer = tf.summary.FileWriter('/cache/tensorboard-logdir/{}'.format(run_name), graph=tf.get_default_graph())

            with open('{}.log'.format(run_name), 'w') as logfile:

                print('Training model {}'.format(run_name))

                sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

                for epoch in range(n_epochs):


                    # For collecting latent dim embeddings
                    z_out = np.zeros((0, dim_z))

                    test_losses = []
                    test_likelihoods = []
                    test_divergences = []
                    train_losses = []
                    train_likelihoods = []
                    train_divergences = []


                    # Loop over all batches
                    for i in range(n_batches_train):
                        # Compute the offset of the current minibatch in the data.
                        offset = i * batch_size
                        batch_train_input = train_data[offset:(offset + batch_size)]

                        _, tot_loss, loss_likelihood, loss_divergence, train_summary, z_in = sess.run(
                            (train_op, loss, neg_marginal_likelihood, KL_divergence, training_summaries, z),
                            feed_dict={x_hat: batch_train_input, x: batch_train_input, keep_prob: 1.0})
                        writer.add_summary(train_summary, epoch * n_batches_train + i)

                        train_losses.append(tot_loss)
                        train_likelihoods.append(loss_likelihood)
                        train_divergences.append(loss_divergence)

                        # Add embeddings to list
                        z_out = np.concatenate((z_out, z_in), axis=0)

                    for i in range(n_batches_test):
                        offset = i * batch_size
                        batch_test_input = test_data[offset:(offset + batch_size)]

                        test_loss, test_likelihood, test_divergence, reconstr, z_in = sess.run(
                            (loss, neg_marginal_likelihood, KL_divergence, y, z),
                            feed_dict={x_hat: batch_test_input, x: batch_test_input, keep_prob: 1.0}
                        )
                        test_losses.append(test_loss)
                        test_likelihoods.append(test_likelihood)
                        test_divergences.append(test_divergence)

                        z_out = np.concatenate((z_out, z_in), axis=0)

                    all_losses += test_losses
                    all_likelihoods += test_likelihoods
                    all_divergences += test_divergences

                    test_summaries = tf.Summary()
                    test_summaries.value.add(tag="test/loss", simple_value=np.mean(test_losses))
                    test_summaries.value.add(tag="test/neg_marginal_likelihood", simple_value=np.mean(test_likelihoods))
                    test_summaries.value.add(tag="test/KL_divergence", simple_value=np.mean(test_divergences))

                    writer.add_summary(test_summaries, (epoch+1) * n_batches_train - 1)

                    # print cost every epoch
                    train_line = "epoch %d: L_tot %.2E L_likelihood %.2E L_divergence %.2E" % (epoch, tot_loss, loss_likelihood, loss_divergence)
                    test_line =  "    test: L_tot %.2E L_likelihood %.2E L_divergence %.2E" % (test_loss, test_likelihood, test_divergence)
                    logfile.write(train_line + '\n')
                    logfile.write(test_line + '\n')
                    print(train_line)
                    print(test_line)

                    # if minimum loss is updated or final epoch, plot results
                    if test_loss < min_tot_loss:
                        min_tot_loss = test_loss
                        #with open('min_{}.txt'.format(run_name), 'w') as log:
                        #    log.write('Minimum total loss: ' + str(test_loss) + '\n')
                        #    log.write('Minimum likelihood: ' + str(test_likelihood) + '\n')
                        #    log.write('Minimum divergence: ' + str(test_divergence) + '\n')

                        #batch_test_input = np.reshape(batch_train_input, (-1, nchannel, 128))
                        #reconstr = np.reshape(reconstr, (-1, nchannel, 128))
                        #for ch in range(nchannel):
                        #    plot_to_file('./images/reconstruction_{}.png'.format(ch),
                        #                 batch_test_input[0][ch], reconstr[0][ch])
                        #    fft = lambda x: np.abs(np.fft.fft(x - np.mean(x)))
                        #    plot_to_file('./images/fft_{}.png'.format(ch),
                        #                 fft(batch_test_input[0][ch]), fft(reconstr[0][ch]))

                    # if training is finished
                    if epoch+1 == n_epochs:
                        #save_embedding(np.concatenate((train_data, test_data)), z_out, sess, writer, channel_idx)

                        save_path = saver.save(sess, '/cache/model-{}.ckpt'.format(run_name))
                        print('Model saved to {}'.format(save_path))

                        #np.save('losses_{}'.format(run_name), all_losses)
                        #np.save('likelihoods_{}'.format(run_name), all_likelihoods)
                        #np.save('divergences_{}'.format(run_name), all_divergences)

                        #moving_average_n = 10
                        #plot_training('{}.png'.format(run_name),
                        #             [np.convolve(x, np.ones((moving_average_n,))/moving_average_n, mode='valid') for x in
                        #                [all_losses, all_likelihoods, all_divergences]],
                        #             ['Total loss', 'Loss likelihood', 'KL divergence'])

                        batch_test_input = np.reshape(batch_train_input, (-1,nchannel,128))
                        reconstr = np.reshape(reconstr, (-1,nchannel,128))
                        for ch in range(nchannel):
                            plot_to_file('./images/reconstruction_{}.png'.format(ch),
                               batch_test_input[0][ch], reconstr[0][ch])
                            fft = lambda x: np.abs(np.fft.fft(x - np.mean(x)))
                            plot_to_file('./images/fft_{}.png'.format(ch),
                               fft(batch_test_input[0][ch]), fft(reconstr[0][ch]))


        # visualize model
        else:
            saver.restore(sess, './logs/model-{}.ckpt'.format(args.run_name))
            with h5py.File('/data/eeg_channels.h5', 'r') as h5f:
                input = [(h5f['one_sec_norm'][args.channel,args.visualize_sample,:] + 1.0) / 2.0]

                latent_space, reconstr, loss_sample = sess.run(
                    (z, y, loss),
                    feed_dict={x_hat: input,
                               x: input,
                               keep_prob: 1.0} )

                plot_to_file('./logs/{}/reconstruction_{}.png'.format(args.channel, args.visualize_sample),
                               input[0], reconstr[0])

                plot_to_file('./logs/{}/freq_{}.png'.format(args.channel, args.visualize_sample),
                               scipy.signal.periodogram(input[0]*256), scipy.signal.periodogram(reconstr[0]*256))

def run_inference(model_file, input, n_hidden, dim_z):

    tf.reset_default_graph()

    # network architecture
    n_hidden = n_hidden
    dim_input = input.shape[1]  # number of samples per segment
    dim_z = dim_z

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_input], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_input], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    #z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')


    y, z, loss, neg_marginal_likelihood, KL_divergence = model.autoencoder(x_hat, x, dim_input, dim_z, n_hidden, keep_prob)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_file)

        return sess.run(
            (z, y, loss),
            feed_dict={x_hat: input,
                       x: input,
                       keep_prob: 1.0} )

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


if __name__ == '__main__':

    # parse arguments
    args = parse_args()

    print(args)

    if args is None:
        exit()

    # main

    train_total_raw, train_total_data, train_overlap_data, test_raw, test_data, test_overlap_data, _, _ = eeg_data.get_eeg_data()

    #train_total_data = np.repeat(train_total_data[np.newaxis,:,:],23,axis=0)
    #test_data = np.repeat(test_data[np.newaxis,:,:],23,axis=0)
    #print(train_total_data.shape)
    #print(test_data.shape)

    # Data format: second x channel x samples

    # Collapse samples into channels
    train_total_data = np.reshape(train_overlap_data,
                                  (train_overlap_data.shape[0], train_overlap_data.shape[1] * train_overlap_data.shape[2]))
    test_data = np.reshape(test_overlap_data, (test_overlap_data.shape[0], test_overlap_data.shape[1] * test_overlap_data.shape[2]))

    #num_instances = 1
    #import random
    #instances = random.sample(range(len(train_total_data)), num_instances)
    #train_total_data = train_total_data[instances]

    print(test_data.shape)
    #train_total_data = train_total_data[0:10]

    if args.compute_path:
        import random
        from scipy.spatial.distance import euclidean

        path_size = 100

        idx = np.random.randint(0, len(train_total_data), 1)[0]
        inputs = train_total_data[idx:(idx + path_size)]
        z, r, l = run_inference(args.model_path, train_total_data[idx:(idx + path_size)], args.n_hidden, args.dim_z)
        distances = [euclidean(l1, l2) for l1, l2 in zip(z[:-1], z[1:])]
        plot_data_single(distances, 'path.png')
        print('Average step size: {}'.format(np.mean(distances)))

        idx = np.random.randint(0, len(train_total_data), path_size)
        inputs = train_total_data[idx]
        z, r, l = run_inference(args.model_path, train_total_data[idx], args.n_hidden, args.dim_z)
        distances = [euclidean(l1, l2) for l1, l2 in zip(z[:-1], z[1:])]
        plot_data_single(distances, filename='random_path.png')
        print('Average step size: {}'.format(np.mean(distances)))

    else:
        main(args, train_total_data, test_data)
