import tensorflow as tf
import numpy as np
from PIL import Image
import os

import utils

class GanModel():

    def __init__(self, output_dir, train_batch_size=32, x_dim=28, y_dim=28, depth=1, noise_dim=64, learn_rate=1e-3):
        self.output_dir = os.path.abspath(output_dir)

        self._c = utils.obj()
        self._c.train_batch_size = train_batch_size
        self._c.x_dim = x_dim
        self._c.y_dim = y_dim
        self._c.depth = depth
        self._c.noise_dim = noise_dim
        self._c.learn_rate = learn_rate
        self._c.flat_dim = x_dim * y_dim * depth

        self.sess = tf.Session()

        # This object holds tensor/op references that are shared between portions of the network
        self._tf = utils.obj()

        self._generator()
        self._discriminator()
        self._loss()

        self.sess.run(tf.global_variables_initializer())

    def _xavier_init(self, size):
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

            w1 = tf.Variable(self._xavier_init([in1, activation_size]), name='weight1')
            b1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias1')

            w2 = tf.Variable(self._xavier_init([activation_size, out_size]), name='weight2')
            b2 = tf.Variable(tf.zeros(shape=[out_size]), name='bias2')

            outs = list()

            for t_in in inputs:
                act = activation(tf.matmul(t_in, w1) + b1)
                t_out = tf.matmul(act, w2) + b2
                outs.append(t_out)

            return (outs, w1, b1, w2, b2)

    def _generator(self):
        with tf.name_scope('generator_activation'):
            g_input = tf.placeholder(tf.float32, shape=[None, self._c.noise_dim], name='input')

            (outputs, w1, b1, w2, b2) = self._weight([g_input], out_size=self._c.flat_dim)
            
            self._tf.g_input = g_input
            self._tf.g_train_vars = [w1, w2, b1, b2]

            t_out = outputs[0]
            out_shape = [-1, self._c.x_dim, self._c.y_dim, self._c.depth]
            self._tf.g_output = tf.nn.sigmoid(tf.reshape(t_out, out_shape))

    def _discriminator(self):
        with tf.name_scope('discriminator_activation'):
            real_input = tf.placeholder(tf.float32, shape=[None, self._c.x_dim, self._c.y_dim, self._c.depth], name='input')
            
            real_flat = tf.reshape(real_input, [-1, self._c.flat_dim])

            fake_input = self._tf.g_output

            fake_flat = tf.reshape(fake_input, [-1, self._c.flat_dim])

            (outputs, w1, b1, w2, b2) = self._weight([real_flat, fake_flat], out_size=1)

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

    def generate_noise(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self._c.train_batch_size

        return np.random.uniform(-1., 1., size=[batch_size, self._c.noise_dim])

    def run(self, image_repo):
        def output_generated_img(img_arr, out_filename):
            img_arr = np.array(img_arr * 255, dtype=np.uint8)

            if self._c.depth == 1:
                # in a 1-channel image, we need to drop the last dimension so it's formatted properly for PIL.Image.fromarray
                img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1]))

                fmt = 'L'
            elif self._c.depth == 3:
                fmt = 'RGB'
            else:
                raise os.error(f'Invalid image depth={self._c.depth}')

            img = Image.fromarray(img_arr, fmt)
            img.save(out_filename)

        def pathto(sub_path):
            return os.path.join(self.output_dir, sub_path)
        
        def nukedir(dir_path):
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    try:
                        os.remove(os.path.join(dir_path, f))
                    except:
                        pass

        gen_images_dir = pathto('gen')
        nukedir(gen_images_dir)

        if not os.path.exists(gen_images_dir):
            os.makedirs(gen_images_dir)
        
        nukedir(pathto('tb'))
        tb_writer = tf.summary.FileWriter(pathto('tb'), self.sess.graph)
        all_summaries_tensor = tf.summary.merge_all()
    
        dataset = image_repo.get_dataset().repeat(20).batch(g._c.train_batch_size)

        ds_iterator = tf.data.make_one_shot_iterator(dataset)
        iter_batch = ds_iterator.get_next()

        global_step = 0

        try:
            while True:
                image_batch = self.sess.run(iter_batch)

                generated_images, g_loss, d_loss, _, _, summary_result = self.sess.run([
                    g._tf.g_output,
                    g._tf.g_loss,
                    g._tf.d_loss,
                    g._tf.g_solver,
                    g._tf.d_solver,
                    all_summaries_tensor
                ], feed_dict={
                    g._tf.d_input: image_batch,
                    g._tf.g_input: g.generate_noise(len(image_batch))
                })

                tb_writer.add_summary(summary_result, global_step=global_step)

                if global_step % 500 == 0:
                    print(f'global_step={global_step}, g_loss={g_loss:.4}, d_loss={d_loss:.4}')
                    
                    out_img_it = 1
                    for gen_img in generated_images:
                        if out_img_it > 3:
                            break
                        outimg_name = pathto(f'gen/{global_step}_{out_img_it}.png')
                        output_generated_img(gen_img, outimg_name)
                        out_img_it += 1

                global_step += 1
        except tf.errors.OutOfRangeError:
            pass

if __name__ == '__main__':
    image_repo = utils.ImageRepository.create_or_open('.data/mnist/cache.tfrecords', '.data/mnist/raw')
    g = GanModel('bin')
    g.run(image_repo)