from tensorflow import keras
from tensorflow.nn import relu

def build_model():
    def conv2d_down(filter_ct):
        return keras.layers.Conv2D(filters=filter_ct, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=relu)

    def deconv2d_up(filter_ct, output_padding=None):
        return keras.layers.Conv2DTranspose(filters=filter_ct, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=relu, output_padding=output_padding)

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(1080, 1920, 3)),

        # in [None, 1080, 1920, 3]
        conv2d_down(8),

        # in [None, 540, 960, 8]
        conv2d_down(16),

        # in [None, 270, 480, 16]
        NestedConcat([
            conv2d_down(32),

            # in [None, 135, 240, 32]
            conv2d_down(48),

            # in [None, 68, 120, 48]
            NestedConcat([
                conv2d_down(64),

                # in [None, 34, 60, 64]
                conv2d_down(80),

                # in [None, 17, 30, 80]
                deconv2d_up(64),

                # in [None, 34, 60, 48]
                deconv2d_up(48)
            ]), # NestedConcat out [None, 68, 120, 96]

            # in [None, 68, 120, 96]
            deconv2d_up(32, output_padding=(0, 1)),

            # in [None, 135, 240, 32]
            deconv2d_up(16)
        ]), # NestedConcat out [None, 270, 480, 32]

        # in [None, 270, 480, 32]
        deconv2d_up(16),

        # in [None, 540, 960, 16]
        deconv2d_up(3)

        # out [None, 1080, 1920, 3]
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

class NestedAdd(keras.layers.Layer):
    # Resnet layer as defined http://torch.ch/blog/2016/02/04/resnets.html

    def __init__(self, inner_layers):
        super(NestedAdd, self).__init__()
        self._inner_layers = inner_layers
        self._add = keras.layers.Add()

    def call(self, inp):
        x = inp
        for y in self._inner_layers:
            x = y(x)
        return self._add([x, inp])

class NestedConcat(keras.layers.Layer):
    # Resnet layer as defined http://torch.ch/blog/2016/02/04/resnets.html

    def __init__(self, inner_layers):
        super(NestedConcat, self).__init__()
        self._inner_layers = inner_layers
        self._concat = keras.layers.Concatenate(axis=3)

    def call(self, inp):
        x = inp
        for y in self._inner_layers:
            x = y(x)
        return self._concat([x, inp])
