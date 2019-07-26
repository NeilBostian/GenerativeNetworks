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

    def _cache_to_file(self, img_dir):
        with tf.Session() as sess:
            def map_fn(t_in):
                return tf.read_file(t_in)

            t_imgname_input = tf.placeholder(shape=[None], dtype=tf.string)
            t_readimgs_output = tf.map_fn(map_fn, t_imgname_input, dtype=tf.string)

            def read_img_list(img_list):
                return sess.run(t_readimgs_output, feed_dict={t_imgname_input: img_list})
            
            def batch_input_dir():
                tmp = list()
                for img_name in listdir_absolute(img_dir):
                    tmp.append(img_name)
                    if len(tmp) >= 128:
                        yield tmp
                        tmp = list()

                if len(tmp) > 0:
                    yield tmp

            with tf.io.TFRecordWriter(self.file_name) as writer:
                def byte_feature(bytes_in):
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_in]))

                for batch in batch_input_dir():
                    for image in read_img_list(batch):
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'img_data': byte_feature(image)
                        }))

                        writer.write(example.SerializeToString())

    def get_dataset(self):
        def decode(serialized_example):
            features = {
                'img_data': tf.FixedLenFeature((), tf.string)
            }

            example = tf.parse_single_example(serialized_example, features=features)

            raw_img_data = example['img_data']
            decoded_img = tf.image.decode_image(raw_img_data)
            return tf.cast(decoded_img, tf.float32) / 255.0

        return tf.data.TFRecordDataset(self.file_name).map(decode)

    def create_or_open(file_name, img_dir):
        ir = ImageRepository(file_name)

        if not ir._exists():
            ir._cache_to_file(img_dir)

        return ir