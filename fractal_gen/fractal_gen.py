import tensorflow as tf
import numpy as np
from PIL import Image

R = 4
ITER_NUM = 300

class FractalGenTensorflowModel():
    def __init__(self, shape_in):
        self._sess = tf.Session()

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
        print('init1')
        g = np.full(shape=Z.shape, fill_value=c, dtype=Z.dtype)
        print('init2')
        xs = tf.Variable(g)
        zs = tf.Variable(Z)
        ns = tf.Variable(tf.zeros_like(xs, tf.float32))

        zs_ = tf.where(tf.abs(zs) < R, zs**2 + xs, zs)
        not_diverged = tf.abs(zs_) < R
        step = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(not_diverged, tf.float32))
        )

        sess.run(tf.global_variables_initializer())

        for i in range(ITER_NUM):
            print(f'\titer{i}')
            sess.run(step)

        final_step = sess.run(ns)
        final_z = sess.run(zs_)

        r, g, b = np.frompyfunc(self._get_color(bg_ratio, ratio), 2, 3)(final_z, final_step)

        img_array = np.uint8(np.dstack((r, g, b)) * 255)

        return Image.fromarray(img_array)

class FractalGenModel():
    def __init__(self):        
        #input params
        ratio = 16.0 / 9.0
        self.start_y = -.7
        self.end_y = .7
        self.start_x = self.start_y * ratio
        self.end_x = self.end_y * ratio
        self.width = 1920 # image width 7680
        self.bg_ratio = (1, 1, 1) # background color ratio
        self.ratio = (0.9, 0.9, 0.9)

        step = (self.end_x - self.start_x) / self.width
        Y, X = np.mgrid[self.start_y:self.end_y:step, self.start_x:self.end_x:step]
        self._Z = X + 1j * Y

        self._tfmodel = FractalGenTensorflowModel(self._Z.shape)

    def generate_sequence(self, iters=300):
        for i in range(0, iters):
            theta = 2 * np.pi / iters * i
            c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
            # c = -0.8 * 1j
            y = self._tfmodel.generate_image(self._Z, c, self.bg_ratio, self.ratio)
            print(f'Fractal gen completed {i+1}/{iters}')
            yield y
