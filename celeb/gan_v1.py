import os
import tensorflow as tf
from utils import obj

imgs_dir = '.celeb_data'
img_shape = [178, 218, 3]

class GanModel():
    def __init__(self, x_dim, y_dim, learn_rate = 1e-3):
        self.sess = tf.Session()

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

        self.sess.run(tf.global_variables_initializer())


    def _log(self, x):
        return tf.log(x + 1e-8)

    def _xavier_init(self, size):
        """
        Xavier Init is used to assign initial values to our weights
        you can read more here: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        """
        with tf.name_scope('xavier_init'):
            in_dim = tf.cast(size[0], tf.float32)
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _weight(self, t_in, activation_size = 128, out_size=-1, activation=tf.nn.relu):
        r = self.sess.run(tf.rank(t_in))
        if r != 1:
            raise os.error(f'input tensor must have rank 1, received rank {r}')

        with tf.name_scope('weight'):
            in_size = self.sess.run(tf.size(t_in))
            if out_size == -1: out_size = in_size

            w1 = tf.Variable(self._xavier_init([in_size, activation_size]), name='weight')
            b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias')
            
            w2 = tf.Variable(self._xavier_init([activation_size, out_size]), name='weight')
            b2 = tf.Variable(tf.zeros(shape=[out_size]), name='bias')

            apply_w1b1 = tf.squeeze(tf.matmul(tf.expand_dims(t_in, 0), w1), 0) + b1
            apply_activation = activation(apply_w1b1)
            apply_w2b2 = tf.squeeze(tf.matmul(tf.expand_dims(apply_activation, 0), w2), 0) + b2
            return (apply_w2b2, [w1, w2], [b1, b2])

    def _fully_connect(self, inputs, num_outputs, name_prefix=None, out_size=-1):
        """
        Fully connects all input channels using matrix dot product
        `inputs` must have length >= 2
        `num_outputs` must be >= 1
        `out_size` is the size of each output node

        Returns: (outputs, weights, biases)
        """
        outputs = list()
        weights = list()
        biases = list()

        name_prefix = name_prefix or 'fc'

        for i in range(num_outputs):
            with tf.name_scope(f'{name_prefix}_{i}'):
                stack = tf.stack(inputs)
                mean = tf.reduce_mean(stack, 0)
                (t_out, wts, bs) = self._weight(mean, out_size=out_size)
                outputs.append(t_out)
                weights.extend(wts)
                biases.extend(bs)

        return (outputs, weights, biases)

    def _phase_1_create_generator_activations(self):
        with tf.name_scope('generator'):
            input = tf.placeholder(tf.float32, shape=self.shape, name='input')

            with tf.name_scope('split_channels'):
                flat_input = tf.reshape(input, [self.flat_dim, self.num_color_channels])
                split_channels = [tf.reshape(x, [self.flat_dim]) for x in tf.split(flat_input, self.num_color_channels, axis=1)]
            
                channels = [tf.reduce_mean(flat_input, 1)] # greyscale as the mean of all 3 color channels
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

                with tf.name_scope(cname):
                    (out_channel, weights, biases) = self._weight(in_channel)

                    out_channels.append(out_channel)
                    all_weights.extend(weights)
                    all_biases.extend(biases)

            (fc_channels, fc_weights, fc_biases) = self._fully_connect(out_channels, self.num_color_channels)

            all_weights.extend(fc_weights)
            all_biases.extend(fc_biases)

            with tf.name_scope('merge_fc'):
                self._tf.g_output = tf.reshape(tf.stack(fc_channels), self.shape)

            self._tf.g_weights = all_weights
            self._tf.g_biases = all_biases

    def _phase_2_create_discriminator_activations(self):
        def discriminate(t_in):
            """
            Takes input tensor of shape `self.shape`, and discriminates whether it is a real image or not
            Returns tuple(t_out, weights, biases)                
            """

            with tf.name_scope('split_channels'):
                flat_input = tf.reshape(input, [self.flat_dim, self.num_color_channels])
                split_channels = [tf.reshape(x, [self.flat_dim]) for x in tf.split(flat_input, self.num_color_channels, axis=1)]
            
                channels = [tf.reduce_mean(flat_input, 1)] # greyscale as the mean of all 3 color channels
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

                with tf.name_scope(cname):
                    (out_channel, weights, biases) = self._weight(in_channel)

                    out_channels.append(out_channel)
                    all_weights.extend(weights)
                    all_biases.extend(biases)

            (fc_channels, fc_weights, fc_biases) = self._fully_connect(out_channels, self.num_color_channels)

            all_weights.extend(fc_weights)
            all_biases.extend(fc_biases)

            (t_out, t_out_weights, t_out_biases) = self._fully_connect(fc_channels, 1, name_prefix='fc_output', out_size=1)
            all_weights.extend(t_out_weights)
            all_biases.extend(t_out_biases)

            return (t_out, all_weights, all_biases)

        with tf.name_scope('discriminator'):
            with tf.name_scope('real'):
                input = tf.placeholder(tf.float32, shape=self.shape, name='input')
                (real, real_weights, real_biases) = discriminate(input)
                self._tf.d_real_output = real
                self._tf.d_real_weights = real_weights
                self._tf.d_real_biases = real_biases
            
            with tf.name_scope('fake'):
                input = self._tf.g_output
                (fake, fake_weights, fake_biases) = discriminate(input)
                self._tf.d_fake_output = fake
                self._tf.d_fake_weights = fake_weights
                self._tf.d_fake_biases = fake_biases
            

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