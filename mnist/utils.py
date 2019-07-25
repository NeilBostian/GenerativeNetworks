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

class ImageRepository():
    def __init__(self, file_name):
        self.file_name = file_name
        self.sess = tf.Session()

    def _exists(self):
        return os.path.exists(self.file_name)

    def _cache_to_file(self, img_dir):
        def read_file(file_name):
            img_raw = tf.image.decode_image(tf.read_file(file_name))
            return tf.cast(img_raw, tf.float32) / 255.0

        input_file_names = tf.placeholder(shape=[None], dtype=tf.string)
        input_file_tensors = tf.map_fn(read_file, input_file_names, dtype=tf.float32)

        images = self.sess.run([input_file_tensors], feed_dict={input_file_names: listdir_absolute(img_dir)})

        with tf.io.TFRecordWriter(self.file_name) as writer:
            def byte_feature(bytes_in):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_in]))

            def int_feature(int_in):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[int_in]))

            for image in images:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int_feature(image.shape[0]),
                    'width': int_feature(image.shape[1]),
                    'img_data': byte_feature(image.tostring())
                }))

                writer.write(example.SerializeToString())

    def get_dataset(self, batch_size=1):
        def decode(serialized_example):
            features = {
                'height': tf.FixedLenFeature((), tf.int64),
                'width': tf.FixedLenFeature((), tf.int64),
                'img_data': tf.FixedLenFeature((), tf.string)
            }

            example = tf.parse_single_example(serialized_example, features=features)

            height = example['height']
            width = example['width']
            raw_img_data = example['img_data']

            img_flat = tf.decode_raw(raw_img_data, out_type=tf.float32)
            return tf.reshape(img_flat, [height, width, -1])

        return tf.data.TFRecordDataset(self.file_name).map(decode)

    def create_or_open(file_name, img_dir):
        ir = ImageRepository(file_name)

        if not ir._exists():
            ir._cache_to_file(img_dir)

        return ir