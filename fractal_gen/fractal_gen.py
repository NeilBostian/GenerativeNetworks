import tensorflow as tf
import numpy as np
from PIL import Image

class FractalGenTensorflowModel():
    def __init__(self, shape_in):
        self._sess = tf.Session()

        self._R = 4
        self._ITER_NUM = 200

        self._build_model(shape_in)

        self._sess.run(tf.global_variables_initializer())

        tf.summary.FileWriter('tb', self._sess.graph)

    def _build_model(self, shape_in):
        self._in_xs = tf.placeholder(shape=shape_in, dtype=tf.float32)
        self._in_zs = tf.placeholder(shape=shape_in, dtype=tf.float32)

        xs = self._in_xs
        zs = self._in_zs
        
        ns = tf.zeros_like(xs)

        def apply_loop(x_in, z_in, n_in):
            b = tf.convert_to_tensor(tf.abs(z_in) < self._R, name='b')
            z_out = tf.where(b, z_in**2 + x_in, z_in)
            n_out = n_in + tf.cast(b, tf.float32)
            return (z_out, n_out)

        for i in range(0, self._ITER_NUM):
            with tf.name_scope(f'iter{i}'):
                (z_out, n_out) = apply_loop(xs, zs, ns)
                zs = z_out
                ns = n_out

        self._out_final_z = zs
        self._out_final_step = ns

    def _get_color(self, bg_ratio, ratio):
        def color(z, i):
            if abs(z) < self._R:
                return 0, 0, 0
            v = np.log2(i + self._R - np.log2(np.log2(abs(z)))) / 5
            if v < 1.0:
                return v**bg_ratio[0], v**bg_ratio[1], v ** bg_ratio[2]
            else:
                v = max(0, 2 - v)
                return v**ratio[0], v**ratio[1], v**ratio[2]
        return color

    def generate_image(self, Z, c, bg_ratio, ratio):
        final_z, final_step = self._sess.run([self._out_final_z, self._out_final_step], feed_dict={
            self._in_xs: np.full(shape=Z.shape, fill_value=c, dtype=Z.dtype),
            self._in_zs: Z,
        })
        
        r, g, b = np.frompyfunc(self._get_color(bg_ratio, ratio), 2, 3)(final_z, final_step)

        img_array = np.dstack((r, g, b))

        return Image.fromarray(np.uint8(img_array * 255))

class FractalGenModel():
    def __init__(self):        
        #input params
        self.start_x = -1.9
        self.end_x = 1.9
        self.start_y = -1.1
        self.end_y = 1.1
        self.width = 600 # image width
        self.bg_ratio = (1, 1, 1) # background color ratio
        self.ratio = (0.9, 0.9, 0.9)

        step = (self.end_x - self.start_x) / self.width
        Y, X = np.mgrid[self.start_y:self.end_y:step, self.start_x:self.end_x:step]
        self._Z = X + 1j * Y

        self._tfmodel = FractalGenTensorflowModel(self._Z.shape)

    def generate_sequence(self, iters=100):
        for i in range(0, iters):
            theta = 2 * np.pi / iters * i
            c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
            y = self._tfmodel.generate_image(self._Z, c, self.bg_ratio, self.ratio)
            print(f'Fractal gen completed {i+1}/{iters}')
            yield y
