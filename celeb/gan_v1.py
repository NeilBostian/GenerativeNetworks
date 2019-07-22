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
            Processes an image channel (color channel or greyscale) with shape `[self.flat_dim]` Rank/dimension of 1.
            Returns: (out_channel, weights, biases)
                out_channel: Tensor of shape `in_channel.shape`
                weights: list of weights applied during processing
                biases: list of biases applied during processing
            """
            with tf.name_scope(scope_name):
                weights = list()
                biases = list()
                return (in_channel * 1, weights, biases)

        def fully_connect(inputs, num_outputs):
            """
            Fully connects all input channels using matrix dot product
            `inputs` must have length >= 2
            `num_outputs` must be >= 1

            Returns: (outputs, weights, biases)
            """
            outputs = list()
            weights = list()
            biases = list()

            for i in range(num_outputs):
                with tf.name_scope('fc_' + str(i)):
                    stack = tf.stack(inputs)
                    mean = tf.reduce_mean(stack, 0)
                    outputs.append(mean)

            return (outputs, weights, biases)

        with tf.name_scope('generator_activation'):
            input = tf.placeholder(tf.float32, shape=self.shape, name='input')

            with tf.name_scope('split_channels'):
                flat_input = tf.reshape(input, [self.flat_dim, self.num_color_channels])
                split_channels = [tf.reshape(x, [self.flat_dim]) for x in tf.split(flat_input, self.num_color_channels, axis=1)]
            
                channels = [tf.reduce_mean(flat_input, 1)] # greyscale channel 0
                channels.extend(split_channels)
            
            out_channels = list()
            all_weights = list()
            all_biases = list()

            for (in_channel, it) in zip(channels, range(len(channels))):
                cname = 'channel_' + str(it)

                if it == 0: cname = 'greyscale'
                elif it == 1: cname = 'red'
                elif it == 2: cname = 'green'
                elif it == 3: cname = 'blue'

                (out_channel, weights, biases) = process_channel(in_channel, cname)

                out_channels.append(out_channel)
                all_weights.extend(weights)
                all_biases.extend(biases)

            (fc_channels, fc_weights, fc_biases) = fully_connect(out_channels, self.num_color_channels)

            with tf.name_scope('merge_fc'):
                transpose = [tf.squeeze(x) for x in fc_channels]
                stacked_output = tf.stack(transpose)
                self._tf.discriminator_output = tf.reshape(stacked_output, self.shape)

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