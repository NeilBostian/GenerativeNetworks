import os
import numpy as np
import tensorflow as tf
import utils
from PIL import Image

class GanModel():
    def __init__(self, x_dim, y_dim, learn_rate = 1e-5):
        self.sess = tf.Session()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.flat_dim = x_dim * y_dim
        self.num_color_channels = 3
        self.learn_rate = learn_rate
        self.shape = [y_dim, x_dim, self.num_color_channels]

        self._tf = utils.obj()

        self._phase_1_create_generator_activations()
        self._phase_2_create_discriminator_activations()
        self._phase_3_loss()

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

    def _fully_connect(self, inputs, num_outputs, name_prefix=None, activation_size=128, activation=tf.nn.relu):
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

        num_inputs = len(inputs)

        input_size = self.sess.run(tf.size(inputs[0]))

        for i in range(num_outputs):
            with tf.name_scope(f'{name_prefix}_{i}'):
                stack = tf.reshape(tf.stack(inputs), [input_size, num_inputs])

                w1 = tf.Variable(self._xavier_init([num_inputs, activation_size]), name='weight')
                b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias')

                act = activation(tf.matmul(stack, w1) + b1)

                w2 = tf.Variable(self._xavier_init([activation_size, 1]), name='weight')
                b2 = tf.Variable(tf.zeros(shape=[1]), name='bias')

                t_out = tf.squeeze(tf.matmul(act, w2) + b2)

                outputs.append(t_out)
                weights.extend([w1, w2])
                biases.extend([b1, b2])

        return (outputs, weights, biases)

    def _dual_fully_connect(self, inputs1, inputs2, num_outputs, name_prefix=None, activation_size=128, activation=tf.nn.relu):
        if len(inputs1) != len(inputs2):
            raise os.error(f'inputs1 and inputs2 must be the same length. received len(inputs1)={len(inputs1)} len(inputs2)={len(inputs2)}')
        
        outputs1 = list()
        outputs2 = list()
        weights = list()
        biases = list()

        name_prefix = name_prefix or 'fc'

        num_inputs = len(inputs1)

        input_size = self.sess.run(tf.size(inputs1[0]))

        for i in range(num_outputs):
            with tf.name_scope(f'{name_prefix}_{i}'):
                stack1 = tf.reshape(tf.stack(inputs1), [input_size, num_inputs])
                stack2 = tf.reshape(tf.stack(inputs2), [input_size, num_inputs])

                w1 = tf.Variable(self._xavier_init([num_inputs, activation_size]), name='weight')
                b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias')

                w2 = tf.Variable(self._xavier_init([activation_size, 1]), name='weight')
                b2 = tf.Variable(tf.zeros(shape=[1]), name='bias')

                def proc(t_in):
                    act = activation(tf.matmul(t_in, w1) + b1)
                    return tf.squeeze(tf.matmul(act, w2) + b2)

                outputs1.append(proc(stack1))
                outputs2.append(proc(stack2))
                weights.extend([w1, w2])
                biases.extend([b1, b2])

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
                self._tf.g_output = tf.cast(255 * tf.reshape(tf.stack(fc_channels), self.shape), tf.uint8)

            self._tf.g_input = input
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
            
            fake_input = tf.cast(self._tf.g_output, tf.float32)
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

            (t_out_real, t_out_fake, t_out_weights, t_out_biases) = self._dual_fully_connect(fc_channels_real, fc_channels_fake, 1, name_prefix='fc_output')
            all_weights.extend(t_out_weights)
            all_biases.extend(t_out_biases)

            with tf.name_scope('prediction'):
                w1 = tf.Variable(self._xavier_init([self.flat_dim, 1]), name='weight')
                b1 = tf.Variable(tf.zeros([1]), name='bias')

                def apply_weight(t_in):
                    return tf.nn.sigmoid(tf.reshape(tf.matmul(t_in, w1) + b1, [1]))

                t_out_real = apply_weight(t_out_real)
                t_out_fake = apply_weight(t_out_fake)

            self._tf.d_input = real_input
            self._tf.d_real_output = t_out_real[0]
            self._tf.d_fake_output = t_out_fake[0]
            self._tf.d_weights = all_weights
            self._tf.d_biases = all_biases

    def _phase_3_loss(self):
        with tf.name_scope('loss'):
            var_list = self._tf.g_weights + self._tf.g_biases + self._tf.d_weights + self._tf.d_biases
            self._tf.loss = self._tf.d_fake_output
            self._tf.solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self._tf.loss, var_list=var_list)

if __name__ == '__main__':
    def sample_z(m, n, p):
        return np.random.uniform(-1., 1., size=[m, n, p])

    if os.path.exists('bin/tb'):
        for f in os.listdir('bin/tb'):
            os.remove('bin/tb/' + f)

    g = GanModel(178, 218)
    tf.summary.FileWriter('bin/tb', g.sess.graph)

    imgs_dir = 'C:\\repos\\_data\\celeb\\imgs'

    if not os.path.exists('bin/gen'):
        os.makedirs('bin/gen')
        
    sess = g.sess
    it = 0
    for img in utils.get_img_feed(utils.listdir_absolute(imgs_dir)):
        gen, a, _ = sess.run([
            g._tf.g_output,
            g._tf.loss,
            g._tf.solver
        ], feed_dict={
            g._tf.g_input: sample_z(218, 178, 3),
            g._tf.d_input: img,
        })

        if it % 25 == 0:
            print(f'it={str(it).zfill(3)}: loss={a}')
            gen = np.ascontiguousarray(gen)
            img = Image.fromarray(gen, 'RGB')
            img.save(f'bin/gen/{str(it).zfill(3)}.png')

        it += 1
