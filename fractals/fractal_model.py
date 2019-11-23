import os
from tensorflow import keras

def build_model():
    def dropout(rate):
        return keras.layers.Dropout(rate)

    def conv2d(filter_ct):
        return keras.layers.Conv2D(filters=filter_ct, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    def pool2d():
        return keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def dense(units):
        return keras.layers.Dense(units, activation='relu')

    def deconv2d(filter_ct, output_padding=None):
        return keras.layers.Conv2DTranspose(filters=filter_ct, kernel_size=(5, 5), strides=(2, 2), padding='same', output_padding=output_padding)

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(1080, 1920, 3)),

        # in [None, 1080, 1920, 3]
        conv2d(64),

        # in [None, 1080, 1920, 64]
        pool2d(),

        # in [None, 540, 960, 64]
        conv2d(128),
        dropout(0.15),

        # in [None, 540, 960, 128]
        pool2d(),

        # in [None, 270, 480, 128]
        conv2d(256),

        # in [None, 270, 480, 256]
        pool2d(),

        # in [None, 135, 240, 256]
        conv2d(512),
        dropout(0.15),

        # in [None, 135, 240, 512]
        pool2d(),

        # in [None, 68, 120, 512]
        conv2d(1024),

        # in [None, 68, 120, 1024]
        pool2d(),
        
        # in [None, 34, 60, 1024]
        conv2d(2048),
        dropout(0.15),

        # in [None, 34, 60, 2048]
        pool2d(),
        
        # in [None, 17, 30, 2048]
        conv2d(4096),

        # in [None, 17, 30, 4096]
        pool2d(),

        # in [None, 9, 15, 4096]
        dense(4096),

        # in [None, 9, 15, 4096]
        dense(2048),
        dropout(0.15),

        # in [None, 9, 15, 2048]
        dense(4096),
        
        # in [None, 9, 15, 4096]
        deconv2d(2048, output_padding=(0, 1)),

        # in [None, 17, 30, 2048]
        deconv2d(1024),
        dropout(0.15),

        # in [None, 34, 60, 1024]
        deconv2d(512),

        # in [None, 68, 120, 512]
        deconv2d(256, output_padding=(0, 1)),

        # in [None, 135, 240, 256]
        deconv2d(128),

        # in [None, 270, 480, 128]
        deconv2d(64),

        # in [None, 540, 960, 64]
        deconv2d(64),

        # in [None, 1080, 1920, 64]
        dense(64),
        dense(3)

        # out [None, 1080, 1920, 3]
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='categorical_crossentropy')

    return model