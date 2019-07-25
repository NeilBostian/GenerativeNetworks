import os
import numpy as np
import tensorflow as tf

def listdir_absolute(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

class obj():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class ImageRepository():
    def __init__(self, file_name):
        self.file_name = file_name

    def _exists(self):
        return os.path.exists(self.file_name)

    def _cache_to_file(self, img_dir, validate_shapes=True):
        # TODO: Update the writing to be a tf op so we don't have to read
        # all images into memory before writing the tfrecords.

        def read_file(file_name):
            img_raw = tf.image.decode_image(tf.read_file(file_name))
            return tf.cast(img_raw, tf.float32) / 255.0

        input_file_names = tf.placeholder(shape=[None], dtype=tf.string)
        input_file_tensors = tf.map_fn(read_file, input_file_names, dtype=tf.float32)

        with tf.Session() as sess:
            images = sess.run(input_file_tensors, feed_dict={input_file_names: listdir_absolute(img_dir)})

        with tf.io.TFRecordWriter(self.file_name) as writer:
            def byte_feature(bytes_in):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_in]))

            def int_feature(int_in):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[int_in]))

            known_height = None
            known_width = None
            known_depth = None

            for image in images:
                height = image.shape[0]
                width = image.shape[1]
                depth = image.shape[2]

                if known_height == None: known_height = height
                elif validate_shapes and known_height != height: raise os.error('Image height did not match previous image')

                if known_width == None: known_width = width
                elif validate_shapes and known_width != width: raise os.error('Image width did not match previous image')

                if known_depth == None: known_depth = depth
                elif validate_shapes and known_depth != depth: raise os.error('Image depth did not match previous image')

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int_feature(height),
                    'width': int_feature(width),
                    'depth': int_feature(depth),
                    'img_data': byte_feature(image.tostring())
                }))

                writer.write(example.SerializeToString())

    def get_dataset(self):
        def decode(serialized_example):
            features = {
                'height': tf.FixedLenFeature((), tf.int64),
                'width': tf.FixedLenFeature((), tf.int64),
                'depth': tf.FixedLenFeature((), tf.int64),
                'img_data': tf.FixedLenFeature((), tf.string)
            }

            example = tf.parse_single_example(serialized_example, features=features)

            height = example['height']
            width = example['width']
            depth = example['depth']
            raw_img_data = example['img_data']

            img_flat = tf.decode_raw(raw_img_data, out_type=tf.float32)
            return tf.reshape(img_flat, [height, width, depth])

        return tf.data.TFRecordDataset(self.file_name).map(decode)

    def create_or_open(file_name, img_dir, validate_shapes=True):
        ir = ImageRepository(file_name)

        if not ir._exists():
            ir._cache_to_file(img_dir)

        return ir