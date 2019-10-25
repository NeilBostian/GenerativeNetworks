import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL

model_weights_path = '.data/fractal_model.data'

def load_image(infilename) :
    img = PIL.Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return (data.astype(dtype="float32") / 255.0)

def save_image(npdata, outfilename):
    npdata = np.asarray(np.clip(npdata * 255, 0, 255), dtype="uint8")
    img = PIL.Image.fromarray(npdata, "RGB")
    img.save(outfilename)

def build_model():
    model = tf.keras.Sequential([
        keras.layers.InputLayer(input_shape=(1081, 1920, 3)),
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        keras.layers.Dense(16),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        keras.layers.Dense(32),
        keras.layers.Dense(3)
        # keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(1081, 1920, 3)),
        # keras.layers.Dense(64, activation='relu'),
        # 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

def train():
    def iterate_train_paths():
        num_imgs = 1
        for i in range(0, num_imgs):
            index = i
            image_path = f'.data/imgs/f-{i}.png'

            next_index = i + 1
            next_image_path = f'.data/imgs/f-{next_index}.png'
            yield (image_path, next_image_path)

    model = build_model()

    x = np.array([load_image(cur) for cur, nxt in iterate_train_paths()])
    y = np.array([load_image(nxt) for cur, nxt in iterate_train_paths()])

    for i in range(0, 100):
        model.fit(x, y, epochs=10)
        print('saving...')
        model.save(model_weights_path)

def apply(src_img, iters=15):
    model = build_model()
    model.load_weights(model_weights_path)

    x = load_image(src_img).reshape([1, 1081, 1920, 3])
    
    for i in range(0, iters):
        y = model.predict(x)
        save_image(y.reshape([1081, 1920, 3]), f'.data/gen/gen_sample{i}.png')
        x = y

if __name__ == '__main__':
    #train()
    apply(".data/gen/original_sample.png")

