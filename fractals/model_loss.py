import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from train_data import TrainData

class ModelLoss():
    def __init__(self):
        pass

    def call(self, y_true, y_pred):
        return tf.losses.mean_squared_error(y_true, y_pred)

    def build_loss_model():
        def conv2d_down(filters):
            return [
                keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization()
            ]

        def conv2d(filters):
            return [
                keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization()
            ]

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(1080, 1920, 3)),

            # in [None, 1080, 1920, 3]
            *conv2d(4),
            *conv2d_down(4),

            # in [None, 540, 960, 4]
            *conv2d(8),
            *conv2d_down(8),

            # in [None, 270, 480, 8]
            *conv2d(16),
            *conv2d_down(16),

            # in [None, 135, 240, 16]
            *conv2d(32),
            *conv2d_down(32),

            # in [None, 68, 120, 32]
            *conv2d(64),
            *conv2d_down(64),

            # in [None, 34, 60, 64]
            *conv2d(128),
            *conv2d_down(128),

            # in [None, 17, 30, 128]
            *conv2d(256),
            *conv2d_down(256),

            # in [None, 9, 15, 256]
            *conv2d(256),
            *conv2d_down(256),

            # in [None, 5, 8, 256]
            *conv2d(256),
            *conv2d_down(256),

            # in [None, 3, 4, 256]
            *conv2d(256),
            *conv2d_down(256),

            # in [None, 2, 2, 256]
            *conv2d(256),
            *conv2d_down(256),

            # in [None, 1, 1, 256]
            keras.layers.Flatten(),
            
            keras.layers.Dense(512),
            keras.layers.ReLU(),
            keras.layers.Dropout(rate=0.30),

            keras.layers.Dense(512),
            keras.layers.ReLU(),
            keras.layers.Dropout(rate=0.30),

            keras.layers.Dense(2),
            keras.layers.Dense(2),
            keras.layers.Softmax()
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.binary_crossentropy)

        return model

    def train_model_loss():
        model = ModelLoss.build_loss_model()

        def train_iter():
            for i in range(0, 8000):
                g = random.randint(0, 1)
                if i == 0:
                    td = TrainData.get_random()
                    x = TrainData.preprocess_pil_image(td.get_train_image())
                    y = np.array([[1., 0.]])
                else:
                    x = ModelLoss.get_random_train_input()
                    y = np.array([[0., 1.]])

                yield (x, y, i)

        def train_iter_batch(batch_size=4):
            xs = []
            ys = []

            for x, y, i in train_iter():
                xs.append(x[0])
                ys.append(y[0])

                if len(xs) >= batch_size:
                    yield (np.array(xs), np.array(ys), i)

                    xs = []
                    ys = []

        for x, y, i in train_iter_batch():
            print(f'{i+1:05}:')
            model.fit(x=x, y=y)

        if not os.path.exists('.data/loss_model_weights'):
            os.mkdir('.data/loss_model_weights')

        model.save_weights('.data/loss_model_weights/model_weights')

    def get_random_train_input():
        if not os.path.exists('.data/DIV2K'):
            # DIV2K Home Page: https://data.vision.ee.ethz.ch/cvl/DIV2K/
            # DIV2K Training Set: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
            raise os.error('No DIV2K Training set found in .data/DIV2K directory. Download http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip')

        img_names = os.listdir('.data/DIV2K')

        def get_rand():
            i = random.randint(0, len(img_names) - 1)
            filename = f'.data/DIV2K/{img_names[i]}'
            img = Image.open(filename)
            img.load()
            if img.height < 1080 or img.width < 1080 or (img.height < 1920 and img.width < 1920):
                img.close()
                return None
            return img
        
        x = get_rand()
        while x == None: x = get_rand()
        return ModelLoss._format_train_image(x)

    def _format_train_image(img):
        data = np.asarray(img, dtype=np.uint8)
        img.close()

        data = data.astype(dtype=np.float32) / 255.0

        height, width, channels = data.shape

        if height > width:
            data = np.transpose(data, (1, 0, 2))
            x = height
            height = width
            width = x

        if height > 1080:
            starty = height // 2 - 540
            endy = starty + 1080
            data = data[starty:endy, :, :]

        if width > 1920:
            startx = width // 2 - 960
            endx = startx + 1920
            data = data[:, startx:endx, :]

        return np.reshape(data, [1, 1080, 1920, 3])

if __name__ == '__main__':
    ModelLoss.train_model_loss()