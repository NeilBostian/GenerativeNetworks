import os
import tensorflow as tf
from utils import obj

imgs_dir = '.celeb_data'
img_shape = [178, 218, 3]

class GanModel():
    def __init__(self, x_dim, y_dim, learn_rate = 1e-3):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.flat_dim = x_dim * y_dim
        self.num_color_channels = 3
        self.learn_rate = learn_rate
        self.shape = [x_dim, y_dim, self.num_color_channels]
        self.flat_channel_shape = [self.flat_dim, self.num_color_channels]

        self._tf = obj()

        self._phase_1_create_generator_activations()
        self._phase_2_create_discriminator_activations()
        self._phase_3_global_loss()
        self._phase_4_generator_loss()
        self._phase_5_discriminator_loss()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.sess = sess


    def _log(self, x):
        return tf.log(x + 1e-8)

    def _xavier_init(self, size):
        """
        Xavier Init is used to assign initial values to our weights
        you can read more here: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        """
        with tf.name_scope('xavier_init'):
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _phase_1_create_generator_activations(self):
        def process_channel(in_channel, scope_name):
            """
            Processes an image channel (color channel or greyscale).
            Returns: (out_channel, weights, biases)
                out_channel: Tensor of shape `in_channel.shape`
                weights: list of weights applied during processing
                biases: list of biases applied during processing
            """
            with tf.name_scope(scope_name):
                weights = list()
                biases = list()
                return (in_channel * 1, weights, biases)


        with tf.name_scope('generator_activation'):
            input = tf.placeholder(tf.float32, shape=self.shape, name='input')

            with tf.name_scope('split_channels'):
                flat_input = tf.reshape(input, [self.flat_dim, self.num_color_channels])
                split_channels = [tf.reshape(x, [self.flat_dim]) for x in tf.split(flat_input, self.num_color_channels, axis=1)]
            
                channels = [tf.reduce_mean(flat_input, 1)] # greyscale channel
                channels.extend(split_channels)
            
            for (in_channel, it) in zip(channels, range(len(channels))):
                cname = 'channel_' + str(it)

                if it == 0: cname = 'greyscale'
                elif it == 1: cname = 'red'
                elif it == 2: cname = 'green'
                elif it == 3: cname = 'blue'

                (out_channel, weights, biases) = process_channel(in_channel, cname)

    def _phase_2_create_discriminator_activations(self):
        pass

    def _phase_3_global_loss(self):
        pass

    def _phase_4_generator_loss(self):
        pass

    def _phase_5_discriminator_loss(self):
        pass

if __name__ == '__main__':
    for f in os.listdir('bin/tb'):
        os.remove('bin/tb/' + f)

    g = GanModel(178, 218)
    tf.summary.FileWriter('bin/tb', g.sess.graph)