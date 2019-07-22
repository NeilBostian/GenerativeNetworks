import os
import tensorflow as tf

def get_img_feed(file_names):
    sess = tf.Session()

    for f in file_names:
        img_raw = tf.read_file(f)
        img_tensor = tf.image.decode_image(img_raw)
        img_unstack = tf.unstack(img_tensor, 3)
        yield sess.run([img_unstack])

def listdir_absolute(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

class obj():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)