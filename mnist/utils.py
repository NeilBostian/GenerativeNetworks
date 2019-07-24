import os
import numpy as np
import tensorflow as tf

def batch_img_feed(dir, batch_size):
    tmp = list()

    sess = tf.Session()

    for f in listdir_absolute(dir):
        img_file = tf.read_file(f)
        img_tensor = tf.cast(tf.image.decode_image(img_file), tf.float32) / 255.0
        tmp.append(img_tensor)
        
        if len(tmp) >= batch_size:
            yield sess.run(tf.stack(tmp))
            tmp = list()

def listdir_absolute(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

class obj():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)