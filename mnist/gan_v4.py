import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import obj

class GanModel():

    def __init__(self, train_batch_size=32, x_dim=28, y_dim=28, channels=1, noise_dim=64, learn_rate=1e-3):
        self._c = obj()
        self._c.train_batch_size = train_batch_size
        self._c.x_dim = x_dim
        self._c.y_dim = y_dim
        self._c.channels = channels
        self._c.noise_dim = noise_dim
        self._c.learn_rate = learn_rate
        self._c.flat_dim = x_dim * y_dim

        self.sess = tf.Session()

        # This object holds tensor/op references that are shared between portions of the network
        self._tf = obj()

        self._generator()
        self._discriminator()
        self._loss()

        self.sess.run(tf.global_variables_initializer())
        

    def _xavier_init_(self, size):
        """
        Xavier Init is used to assign initial values to our weights
        you can read more here: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        """
        with tf.name_scope('xavier_init'):
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _weight(self, inputs, activation_size=128, activation=tf.nn.relu, out_size=-1):
        """
        Weighs and activates a set of input matrices. All inputs must be rank 2, and share the same dimensions.

        For example, input matrix of size (N x M) is weighed like so:
        1. a = (N x M) * weight(M x activation_size) + bias(N x activation_size) -> a has shape (N x activation_size)
        2. b = activation(a) -> b has shape (N x activation_size)
        3. c = b * weight(activation_size x out_size) + bias(N x out_size) -> c has shape (N x out_size)
        4. return c

        @param inputs: list of input tensors to apply the set of weights to
        @param activation_size: the intermediary size our inputs are activated with
        @param activation: the activation function
        @param out_size: the size of output dimension 1. If -1, this is defaulted to the same as input
        @returns: tuple(
            outputs: list of output tensors. Each output tensor corresponds to the input tensor at the same position within the list
            weight1: Tensor for weight 1
            bias1: Tensor for bias 1
            weight2: Tensor for weight 2
            bias2: Tensor for bias 2
        )
        """
        with tf.name_scope('weight'):
            in1 = inputs[0].shape[1].value

            if out_size == -1:
                out_size = in1

            w1 = tf.Variable(self._xavier_init_([in1, activation_size]), name='weight1')
            b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias1')

            w2 = tf.Variable(self._xavier_init_([activation_size, out_size]), name='weight2')
            b2 = tf.Variable(tf.zeros(shape=[out_size]), name='bias2')

            outs = list()

            for t_in in inputs:
                act = activation(tf.matmul(t_in, w1) + b1)
                t_out = tf.matmul(act, w2) + b2
                outs.append(t_out)

            return (outs, w1, b1, w2, b2)

    def _generator(self):
        with tf.name_scope('generator_activation'):
            input = tf.placeholder(tf.float32, shape=[None, self._c.noise_dim], name='input')

            (outputs, w1, b1, w2, b2) = self._weight([input], out_size=self._c.flat_dim)

            self._tf.g_input = input
            self._tf.g_train_vars = [w1, w2, b1, b2]
            self._tf.g_output = tf.nn.sigmoid(outputs[0])

    def _discriminator(self):
        """
            Creates our discriminator model
        """
        with tf.name_scope('discriminator_activation'):
            real_input = tf.placeholder(tf.float32, shape=[None, self._c.flat_dim], name='input')
            fake_input = self._tf.g_output            

            (outputs, w1, b1, w2, b2) = self._weight([real_input, fake_input], out_size=1)

            self._tf.d_input = real_input
            self._tf.d_train_vars = [w1, w2, b1, b2]
            self._tf.d_output_real = outputs[0]
            self._tf.d_output_fake = outputs[1]

    def _loss(self):
        def log(x):
            return tf.log(x + 1e-8)

        with tf.name_scope('loss'):
            d_real = self._tf.d_output_real
            d_fake = self._tf.d_output_fake

            with tf.name_scope('cross'):
                cross = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake))
            
            with tf.name_scope('generator'):
                g_target = 1./(self._c.train_batch_size*2)
                self._tf.g_loss = tf.reduce_sum(g_target * d_real) + tf.reduce_sum(g_target * d_fake) + log(cross)
                self._tf.g_solver = tf.train.AdamOptimizer(learning_rate=self._c.learn_rate).minimize(self._tf.g_loss, var_list=self._tf.g_train_vars)
                
                tf.summary.scalar('loss', self._tf.g_loss)

            with tf.name_scope('discriminator'):
                d_target = 1./self._c.train_batch_size
                self._tf.d_loss = tf.reduce_sum(d_target * d_real) + log(cross)
                self._tf.d_solver = tf.train.AdamOptimizer(learning_rate=self._c.learn_rate).minimize(self._tf.d_loss, var_list=self._tf.d_train_vars)

    def run_training_demo(self):
        def plot(samples):
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            return fig

        def sample_z(m, n):
            return np.random.uniform(-1., 1., size=[m, n])

        sess = self.sess

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter('bin/tb', sess.graph)
        
        if not os.path.exists('bin/gen'):
            os.makedirs('bin/gen')

        i = 0

        mnist = input_data.read_data_sets('.MNIST_data', one_hot=True)

        for it in range(40000):
            d_input, _ = mnist.train.next_batch(self._c.train_batch_size)
            g_input = sample_z(self._c.train_batch_size, self._c.noise_dim)

            d_solver = self._tf.d_solver
            d_loss = self._tf.d_loss

            g_solver = self._tf.g_solver
            g_loss = self._tf.g_loss

            _, D_loss_curr, _, G_loss_curr, summary = sess.run(
                [d_solver, d_loss, g_solver, g_loss, summaries], feed_dict={
                    self._tf.d_input: d_input,
                    self._tf.g_input: g_input
                }
            )

            writer.add_summary(summary, global_step=it)

            if it % 1000 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                    .format(it, D_loss_curr, G_loss_curr))

                genSamples = self._tf.g_output
                samples = sess.run(genSamples, feed_dict={self._tf.g_input: sample_z(16, self._c.noise_dim)})

                fig = plot(samples)
                plt.savefig(f'bin/gen/{str(i).zfill(3)}.png', bbox_inches='tight')
                i += 1
        plt.close(fig)


if __name__ == '__main__':
    g = GanModel()
    g.run_training_demo()