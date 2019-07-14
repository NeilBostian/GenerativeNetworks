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

        self._tensorGlobals = obj()

        self._phase_1_create_generator_activations_()
        self._phase_2_create_discriminator_activations_()
        self._phase_3_global_loss_()
        self._phase_4_generator_loss_()
        self._phase_5_discriminator_loss_()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.sess = sess



    def _log_(self, x):
        return tf.log(x + 1e-8)

    def _xavier_init_(self, size):
        """
        I haven't figured out exactly what this is used for yet
        """
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _phase_1_create_generator_activations_(self):
        with tf.name_scope('generator_activation'):
            g_var1 = tf.placeholder(tf.float32, shape=[None, self._c_.z_dim], name='input')
            w1 = tf.Variable(self._xavier_init_([self._c_.z_dim, self._c_.h_dim]), name='weight1')
            w2 = tf.Variable(self._xavier_init_([self._c_.h_dim, self._c_.x_dim]), name='weight2')
            b1 = tf.Variable(tf.zeros(shape=[self._c_.h_dim]), name='bias1')
            b2 = tf.Variable(tf.zeros(shape=[self._c_.x_dim]), name='bias2')
            theta = [w1, w2, b1, b2]

            def apply_activation(in_tensor):
                intermediary_1 = tf.nn.relu(tf.matmul(in_tensor, w1) + b1)
                intermediary_2 = tf.matmul(intermediary_1, w2) + b2
                intermediary_3 = tf.nn.sigmoid(intermediary_2)
                return intermediary_3

            activation = apply_activation(g_var1)

            self._tensorGlobals.generator_var1 = g_var1
            self._tensorGlobals.generator_weight_1 = w1
            self._tensorGlobals.generator_weight_2 = w2
            self._tensorGlobals.generator_bias_1 = b1
            self._tensorGlobals.generator_bias_2 = b2
            self._tensorGlobals.generator_theta = theta        
            self._tensorGlobals.generator_activation = activation

    def _phase_2_create_discriminator_activations_(self):
        """
            Creates our discriminator model
        """
        with tf.name_scope('discriminator_activation'):
            d_var1 = tf.placeholder(tf.float32, shape=[None, self._c_.x_dim], name='input')
            
            w1 = tf.Variable(self._xavier_init_([self._c_.x_dim, self._c_.h_dim]), name='weight1')
            w2 = tf.Variable(self._xavier_init_([self._c_.h_dim, 1]), name='weight2')
            b1 = tf.Variable(tf.zeros(shape=[self._c_.h_dim]), name='bias1')
            b2 = tf.Variable(tf.zeros(shape=[1]), name='bias2')
            theta = [w1, w2, b1, b2]
            
            def apply_activation(in_tensor):
                activation = tf.nn.relu(tf.matmul(in_tensor, w1) + b1)
                return tf.matmul(activation, w2) + b2

            real = apply_activation(d_var1)
            fake = apply_activation(self._tensorGlobals.generator_activation)

            self._tensorGlobals.discriminator_var1 = d_var1
            self._tensorGlobals.discriminator_weights_1 = w1
            self._tensorGlobals.discriminator_weights_2 = w2
            self._tensorGlobals.discriminator_bias_1 = b1
            self._tensorGlobals.discriminator_bias_2 = b2
            self._tensorGlobals.discriminator_theta = theta
            self._tensorGlobals.discriminator_activation_real = real
            self._tensorGlobals.discriminator_activation_fake = fake

    def _phase_3_global_loss_(self):
        d_real = self._tensorGlobals.discriminator_activation_real
        d_fake = self._tensorGlobals.discriminator_activation_fake

        with tf.name_scope('global_loss'):
            unk_loss = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake))

            self._tensorGlobals.unknown_global_loss = unk_loss

    def _phase_4_generator_loss_(self):
        d_fake = self._tensorGlobals.discriminator_activation_fake
        d_real = self._tensorGlobals.discriminator_activation_real
        theta = self._tensorGlobals.generator_theta
        unk_loss = self._tensorGlobals.unknown_global_loss

        with tf.name_scope('generator_loss'):
            g_target = 1./(self._c_.mb_size*2)
            g_loss = tf.reduce_sum(g_target * d_real) + tf.reduce_sum(g_target * d_fake) + self._log_(unk_loss)
            g_solver = tf.train.AdamOptimizer(learning_rate=self._c_.learn_rate).minimize(g_loss, var_list=theta)

            tf.summary.scalar('loss', g_loss)

            self._tensorGlobals.generator_loss = g_loss
            self._tensorGlobals.generator_solver = g_solver

    def _phase_5_discriminator_loss_(self):
        d_fake = self._tensorGlobals.discriminator_activation_fake
        d_real = self._tensorGlobals.discriminator_activation_real
        theta = self._tensorGlobals.discriminator_theta
        unk_loss = self._tensorGlobals.unknown_global_loss

        with tf.name_scope('discriminator_loss'):
            d_target = 1./self._c_.mb_size
            d_loss = tf.reduce_sum(d_target * d_real) + self._log_(unk_loss)
            d_solver = tf.train.AdamOptimizer(learning_rate=self._c_.learn_rate).minimize(d_loss, var_list=theta)
            
            self._tensorGlobals.discriminator_loss = d_loss
            self._tensorGlobals.discriminator_solver = d_solver


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
        writer = tf.summary.FileWriter('./tb', sess.graph)
        
        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0

        mnist = input_data.read_data_sets('.MNIST_data', one_hot=True)

        for it in range(40000):
            X_mb, _ = mnist.train.next_batch(self._c_.mb_size)
            z_mb = sample_z(self._c_.mb_size, self._c_.z_dim)

            d_solver = self._tensorGlobals.discriminator_solver
            d_loss = self._tensorGlobals.discriminator_loss

            g_solver = self._tensorGlobals.generator_solver
            g_loss = self._tensorGlobals.generator_loss

            X = self._tensorGlobals.discriminator_var1
            z = self._tensorGlobals.generator_var1
            
            _, D_loss_curr = sess.run(
                [d_solver, d_loss], feed_dict={X: X_mb, z: z_mb}
            )

            _, G_loss_curr, summary = sess.run(
                [g_solver, g_loss, summaries], feed_dict={X: X_mb, z: z_mb}
            )

            writer.add_summary(summary, global_step=it)

            if it % 1000 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                    .format(it, D_loss_curr, G_loss_curr))

                genSamples = self._tensorGlobals.generator_activation
                samples = sess.run(genSamples, feed_dict={z: sample_z(16, self._c_.z_dim)})

                fig = plot(samples)
                plt.savefig(f'out/{str(i).zfill(3)}.png', bbox_inches='tight')
                i += 1
        plt.close(fig)


if __name__ == '__main__':
    g = GanModel()
    g.run_training_demo()