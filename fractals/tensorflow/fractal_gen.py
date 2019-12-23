from datetime import datetime
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
        with tf.device('/cpu:0'):
            self._xs = tf.placeholder(shape=[], dtype=dtype_in, name='xs')
            self._zs = tf.placeholder(shape=shape_in, dtype=dtype_in, name='zs')

            xs = tf.fill(shape_in, self._xs)
            zs = self._zs
            ns = tf.zeros(shape=shape_in, dtype=tf.float32)

            for i in range(0, ITER_NUM):
                zs_next = tf.where(tf.abs(zs) < R, zs**2 + xs, zs)
                not_diverged = tf.abs(zs_next) < R
                ns_next = ns + tf.cast(not_diverged, tf.float32)
                zs = zs_next
                ns = ns_next

            self._zs_next = zs
            self._ns_next = ns

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
        with tf.device('/cpu:0'):
            sess = self._sess

            final_z, final_step = sess.run([
                self._zs_next,
                self._ns_next
            ], feed_dict={
                self._xs: c,
                self._zs: Z
            })

            r, g, b = np.frompyfunc(self._get_color(bg_ratio, ratio), 2, 3)(final_z, final_step)

            img_array = np.uint8(np.dstack((r, g, b)) * 255)

            return Image.fromarray(img_array)
