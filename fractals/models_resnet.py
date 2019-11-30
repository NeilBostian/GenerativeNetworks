from tensorflow import keras
from tensorflow.nn import relu

def build_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(1080, 1920, 3)),
        ResConv2D()
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

class ResConv2D(keras.layers.Layer):
    # Resnet layer as defined http://torch.ch/blog/2016/02/04/resnets.html

    def __init__(self):
        super(ResConv2D, self).__init__()
        self._conv1 = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self._batchnorm1 = keras.layers.BatchNormalization()
        self._conv2 = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self._batchnorm2 = keras.layers.BatchNormalization()

    def call(self, inp):
        res1 = self._conv1(inp)
        res2 = self._batchnorm1(res1)
        res3 = relu(res2)
        res4 = self._conv2(res3)
        res5 = self._batchnorm2(res4)
        res6 = inp + res5
        return relu(res6)