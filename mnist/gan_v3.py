import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import obj

class GanModel():

    def __init__(self):
        self._c_ = obj(
            mb_size = 32,
            x_dim = 784,
            z_dim = 64,
            h_dim = 128,
            learn_rate = 1e-3
        )

        self._tf = obj()

        self._generator()
        self._discriminator()
        self._loss()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.sess = sess

    def _xavier_init_(self, size):
        """
        I haven't figured out exactly what this is used for yet
        """
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _generator(self):
        with tf.name_scope('generator_activation'):
            input = tf.placeholder(tf.float32, shape=[None, self._c_.z_dim], name='input')
            w1 = tf.Variable(self._xavier_init_([self._c_.z_dim, self._c_.h_dim]), name='weight1')
            w2 = tf.Variable(self._xavier_init_([self._c_.h_dim, self._c_.x_dim]), name='weight2')
            b1 = tf.Variable(tf.zeros(shape=[self._c_.h_dim]), name='bias1')
            b2 = tf.Variable(tf.zeros(shape=[self._c_.x_dim]), name='bias2')
            train_vars = [w1, w2, b1, b2]

            def apply_activation(in_tensor):
                intermediary_1 = tf.nn.relu(tf.matmul(in_tensor, w1) + b1)
                intermediary_2 = tf.matmul(intermediary_1, w2) + b2
                intermediary_3 = tf.nn.sigmoid(intermediary_2)
                return intermediary_3

            activation = apply_activation(input)

            self._tf.g_input = input
            self._tf.g_train_vars = train_vars
            self._tf.g_output = activation

    def _discriminator(self):
        """
            Creates our discriminator model
        """
        with tf.name_scope('discriminator_activation'):
            input = tf.placeholder(tf.float32, shape=[None, self._c_.x_dim], name='input')
            
            w1 = tf.Variable(self._xavier_init_([self._c_.x_dim, self._c_.h_dim]), name='weight1')
            w2 = tf.Variable(self._xavier_init_([self._c_.h_dim, 1]), name='weight2')
            b1 = tf.Variable(tf.zeros(shape=[self._c_.h_dim]), name='bias1')
            b2 = tf.Variable(tf.zeros(shape=[1]), name='bias2')
            train_vars = [w1, w2, b1, b2]
            
            def apply_activation(in_tensor):
                activation = tf.nn.relu(tf.matmul(in_tensor, w1) + b1)
                return tf.matmul(activation, w2) + b2

            real = apply_activation(input)
            fake = apply_activation(self._tf.g_output)

            self._tf.d_input = input
            self._tf.d_train_vars = train_vars
            self._tf.d_output_real = real
            self._tf.d_output_fake = fake

    def _loss(self):
        def log(x):
            return tf.log(x + 1e-8)

        with tf.name_scope('loss'):
            d_real = self._tf.d_output_real
            d_fake = self._tf.d_output_fake

            with tf.name_scope('cross'):
                cross = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake))
            
            with tf.name_scope('generator'):
                g_target = 1./(self._c_.mb_size*2)
                self._tf.g_loss = tf.reduce_sum(g_target * d_real) + tf.reduce_sum(g_target * d_fake) + log(cross)
                self._tf.g_solver = tf.train.AdamOptimizer(learning_rate=self._c_.learn_rate).minimize(self._tf.g_loss, var_list=self._tf.g_train_vars)
                
                tf.summary.scalar('loss', self._tf.g_loss)

            with tf.name_scope('discriminator'):
                d_target = 1./self._c_.mb_size
                self._tf.d_loss = tf.reduce_sum(d_target * d_real) + log(cross)
                self._tf.d_solver = tf.train.AdamOptimizer(learning_rate=self._c_.learn_rate).minimize(self._tf.d_loss, var_list=self._tf.d_train_vars)

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
            d_input, _ = mnist.train.next_batch(self._c_.mb_size)
            g_input = sample_z(self._c_.mb_size, self._c_.z_dim)

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
                samples = sess.run(genSamples, feed_dict={self._tf.g_input: sample_z(16, self._c_.z_dim)})

                fig = plot(samples)
                plt.savefig(f'bin/gen/{str(i).zfill(3)}.png', bbox_inches='tight')
                i += 1
        plt.close(fig)


if __name__ == '__main__':
    g = GanModel()
    g.run_training_demo()