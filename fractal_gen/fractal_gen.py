import tensorflow as tf
import numpy as np
from PIL import Image

R = 4
ITER_NUM = 300

class FractalGenTensorflowModel():
    def __init__(self, shape_in, dtype_in):
        self._sess = tf.Session()

        self._build_model(shape_in, dtype_in)

        self._sess.run(tf.global_variables_initializer())

    def _build_model(self, shape_in, dtype_in):
        self._xs = tf.placeholder(shape=shape_in, dtype=dtype_in, name='xs')
        self._zs = tf.placeholder(shape=shape_in, dtype=dtype_in, name='zs')
        self._ns = tf.placeholder(shape=shape_in, dtype=tf.float32, name='ns')

        xs = self._xs
        zs = self._zs
        ns = self._ns

        zs_next = tf.where(tf.abs(zs) < R, zs**2 + xs, zs)
        not_diverged = tf.abs(zs_next) < R
        ns_next = ns + tf.cast(not_diverged, tf.float32)

        self._zs_next = zs_next
        self._ns_next = ns_next

    def _get_color(self, bg_ratio, ratio):
        def color(z, i):
            if abs(z) < R:
                return 0, 0, 0
            v = np.log2(i + R - np.log2(np.log2(abs(z)))) / 5
            if v < 1.0:
                return v**bg_ratio[0], v**bg_ratio[1], v ** bg_ratio[2]
            else:
                v = max(0, 2 - v)
                return v**ratio[0], v**ratio[1], v**ratio[2]
        return color

    def generate_image(self, Z, c, bg_ratio, ratio):
        sess = self._sess

        xs = sess.run(tf.fill(Z.shape, c))
        
        zs_cur = Z
        ns_cur = sess.run(tf.zeros(shape=Z.shape, dtype=tf.float32))

        for i in range(ITER_NUM):
            zs_next, ns_next = sess.run([
                self._zs_next,
                self._ns_next
            ], feed_dict={
                self._xs: xs,
                self._zs: zs_cur,
                self._ns: ns_cur
            })
            zs_cur = zs_next
            ns_cur = ns_next

        r, g, b = np.frompyfunc(self._get_color(bg_ratio, ratio), 2, 3)(zs_cur, ns_cur)

        img_array = np.uint8(np.dstack((r, g, b)) * 255)

        return Image.fromarray(img_array)

class FractalGenModel():
    def __init__(self):        
        #input params
        ratio = 16.0 / 9.0
        self.start_y = -1.3
        self.end_y = self.start_y * -1.
        self.start_x = self.start_y * ratio
        self.end_x = self.end_y * ratio
        self.width = 1920 # image width 7680
        self.bg_ratio = (4, 2.5, 1) # background color ratio
        self.ratio = (0.9, 0.9, 0.9)

        step = (self.end_x - self.start_x) / self.width
        Y, X = np.mgrid[self.start_y:self.end_y:step, self.start_x:self.end_x:step]
        self._Z = X + 1j * Y

        self._tfmodel = FractalGenTensorflowModel(self._Z.shape, self._Z.dtype)

    def generate_sequence(self, iters=3000):
        for i in range(0, iters):
            theta = 2 * np.pi * i / iters
            # c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
            # c = -0.8 * 1j
            c = -(0.835 - 0.05 * np.cos(theta)) - (0.2321 + 0.05 * np.sin(theta)) * 1j
            y = self._tfmodel.generate_image(self._Z, c, self.bg_ratio, self.ratio)
            print(f'Fractal gen completed {i+1}/{iters}')
            yield y
