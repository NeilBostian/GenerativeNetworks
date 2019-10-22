import os
import tensorflow as tf
from tensorflow import keras
import PIL

from image_repository import ImageRepository

def build_model():
    model = tf.keras.Sequential([
        keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(1081, 1920, 3)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='relu')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy')

    return model

def get_images():
    dirname = '.data/imgs_rn/'

    files = [dirname + str(x) for x in sorted([int(y) for y in os.listdir(dirname)])]

    files_shifted = files[1:] + [files[0]]

    for i in range(0, len(files)):
        yield (files[i], files_shifted[i])

def create_model():
    def read_img(sess, filename):
        print('fname:======')
        print(filename)
        filename = tf.placeholder(dtype=tf.string)
        filebytes = tf.read_file(filename)
        raw_img = tf.image.decode_png(filebytes)
        as_float = tf.reshape(tf.cast(raw_img, tf.float32) / 255.0, shape=[1081, 1920, 3])

        return sess.run(as_float, feed_dict={filename: filename})

    model = build_model()
    sess = keras.backend.get_session()

    feed_logits = []
    feed_labels = []

    for curFilename, nxtFilename in get_images():
        feed_logits.append(curFilename)
        feed_labels.append(nxtFilename)

    data = tf.data.Dataset.from_generator(get_images, output_types=tf.string).repeat(10).shuffle(100).batch(16)

    ds_iterator = tf.data.make_one_shot_iterator(data)
    iter_batch = ds_iterator.get_next()

    try:
        while True:
            fnames_batch = sess.run(iter_batch)
            
            imgs = []
            labels = []

            for img, label in fnames_batch:
                imgs.append(read_img(sess, str(img)))
                labels.append(read_img(sess, str(label)))

            model.fit(imgs, labels)
    except tf.errors.OutOfRangeError:
        pass

    model.save('.data/fractal_model.data')

if __name__ == '__main__':
    create_model()