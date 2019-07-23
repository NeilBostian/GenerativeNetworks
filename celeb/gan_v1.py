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

    def _weight(self, t_in, activation_size=128, out_size=-1, activation=tf.nn.relu):
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

    def _dual_weight(self, t_in1, t_in2, activation_size=128, out_size=-1, activation=tf.nn.relu):
        r = self.sess.run(tf.rank(t_in1))
        if r != 1:
            raise os.error(f'input tensor t_in1 must have rank 1, received rank {r}')

        r = self.sess.run(tf.rank(t_in2))
        if r != 1:
            raise os.error(f'input tensor t_in2 must have rank 1, received rank {r}')

        in_size1 = self.sess.run(tf.size(t_in1))
        in_size2 = self.sess.run(tf.size(t_in2))

        if in_size1 != in_size2:
            raise os.error(f't_in1 and t_in2 must have the same size. received tf.size(t_in1)={in_size1}, tf.size(t_in2)={in_size2}')

        in_size = in_size1

        with tf.name_scope('dual_weight'):
            if out_size == -1: out_size = in_size

            w1 = tf.Variable(self._xavier_init([in_size, activation_size]), name='weight')
            b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias')
            
            w2 = tf.Variable(self._xavier_init([activation_size, out_size]), name='weight')
            b2 = tf.Variable(tf.zeros(shape=[out_size]), name='bias')

            def apply_activation(t_in):
                apply_w1b1 = tf.squeeze(tf.matmul(tf.expand_dims(t_in, 0), w1), 0) + b1
                apply_activation = activation(apply_w1b1)
                return tf.squeeze(tf.matmul(tf.expand_dims(apply_activation, 0), w2), 0) + b2
            
            t_out1 = apply_activation(t_in1)
            t_out2 = apply_activation(t_in2)

            return (t_out1, t_out2, [w1, w2], [b1, b2])

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

    def _dual_fully_connect(self, inputs1, inputs2, num_outputs, name_prefix=None, out_size=-1):
        if len(inputs1) != len(inputs2):
            raise os.error(f'inputs1 and inputs2 must be the same length. received len(inputs1)={len(inputs1)} len(inputs2)={len(inputs2)}')

        outputs1 = list()
        outputs2 = list()
        weights = list()
        biases = list()

        name_prefix = name_prefix or 'fc'

        with tf.name_scope('stack_mean1'):
            stack1 = tf.stack(inputs1)
            mean1 = tf.reduce_mean(stack1, 0)

        with tf.name_scope('stack_mean2'):
            stack2 = tf.stack(inputs2)
            mean2 = tf.reduce_mean(stack2, 0)

        for i in range(num_outputs):
            with tf.name_scope(f'{name_prefix}_{i}'):
                (t_out1, t_out2, w, b) = self._dual_weight(mean1, mean2, out_size=out_size)
                outputs1.append(t_out1)
                outputs2.append(t_out2)
                weights.extend(w)
                biases.extend(b)

        return (outputs1, outputs2, weights, biases)


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
        all_weights = list()
        all_biases = list()

        def split(t_in):
            with tf.name_scope('split_channels'):
                flat_input = tf.reshape(t_in, [self.flat_dim, self.num_color_channels])
                split_channels = [tf.reshape(x, [self.flat_dim]) for x in tf.split(flat_input, self.num_color_channels, axis=1)]
            
                channels = [tf.reduce_mean(flat_input, 1)] # greyscale as the mean of all 3 color channels
                channels.extend(split_channels)
                return channels

        with tf.name_scope('discriminator'):
            real_input = tf.placeholder(tf.float32, shape=self.shape, name='input')
            real_channels = split(real_input)
            
            fake_input = self._tf.g_output
            fake_channels = split(fake_input)

            real_out_channels = list()
            fake_out_channels = list()

            for (it, (real_channel, fake_channel)) in zip(range(len(real_channels)), zip(real_channels, fake_channels)):
                cname = 'channel_' + str(it)

                if it == 0: cname = 'greyscale'
                elif it == 1: cname = 'red'
                elif it == 2: cname = 'green'
                elif it == 3: cname = 'blue'

                with tf.name_scope(cname):
                    (out_channel_real, out_channel_fake, weights, biases) = self._dual_weight(real_channel, fake_channel)

                    real_out_channels.append(out_channel_real)
                    fake_out_channels.append(out_channel_fake)
                    all_weights.extend(weights)
                    all_biases.extend(biases)

            (fc_channels_real, fc_channels_fake, fc_weights, fc_biases) = self._dual_fully_connect(real_out_channels, fake_out_channels, self.num_color_channels)

            all_weights.extend(fc_weights)
            all_biases.extend(fc_biases)

            (t_out_real, t_out_fake, t_out_weights, t_out_biases) = self._dual_fully_connect(fc_channels_real, fc_channels_fake, 1, name_prefix='fc_output', out_size=1)
            all_weights.extend(t_out_weights)
            all_biases.extend(t_out_biases)

            self._tf.d_real_output = t_out_real
            self._tf.d_fake_output = t_out_fake            

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