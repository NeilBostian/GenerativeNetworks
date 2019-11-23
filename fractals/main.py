import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import PIL

from train_data import TrainData
from fractal_gen import FractalGenTensorflowModel
from fractal_model import build_model

tensorboard_dir = '.data\\tensorboard'
train_checkpoints_dir = '.data/model_checkpoints'
sample_inputs_dir = '.data/model_sample_inputs'
sample_outputs_dir = '.data/model_samples'

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

if not os.path.exists('.data'):
    os.mkdir('.data')

if not os.path.exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)

if not os.path.exists(train_checkpoints_dir):
    os.mkdir(train_checkpoints_dir)

if not os.path.exists(sample_inputs_dir):
    os.mkdir(sample_inputs_dir)

if not os.path.exists(sample_outputs_dir):
    os.mkdir(sample_outputs_dir)

def main():
    model = build_model()

    last_checkpoint = get_latest_model_checkpoint()

    if last_checkpoint:
        model.load_weights(f'{train_checkpoints_dir}/{last_checkpoint}')

    current_checkpoint = last_checkpoint or 1

    while True:
        process_checkpoint(model, current_checkpoint)
        current_checkpoint += 1

def get_latest_model_checkpoint():
    ld = list(os.listdir(train_checkpoints_dir))
    ld.sort()

    if len(ld) > 0:
        return int(ld[-1])

def process_checkpoint(model, checkpoint):
    td = TrainData.get_random()
    feature = preprocess_pil_image(td.get_train_image())
    label = preprocess_pil_image(td.get_next_train_image())

    model.fit(feature, label, callbacks=[tb_callback])

    if checkpoint % 5 == 0:
        model.save_weights(f'{train_checkpoints_dir}/{checkpoint}')

    if checkpoint % 100 == 0:
        process_sample_images(model, checkpoint)

def process_sample_images(model, checkpoint):
    """ processes images in the '.data/model_sample_inputs' directory through the model, each with 5 samples """

    for img in os.listdir(sample_inputs_dir):
        try:
            x = PIL.Image.load(f'{sample_inputs_dir}/{img}')
            x = preprocess_pil_image(x)

            for i in range(1, 6):
                x = model.predict(x)

                out_dir = f'{sample_outputs_dir}/{img}'
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                y = postprocess_pil_image(x)
                y.save(f'{out_dir}/{i}.png')
                y.close()
        except:
            pass

def preprocess_pil_image(img):
    """ Preprocess PIL image into the input data type for our keras model """
    data = np.asarray(img, dtype=np.uint8)
    data = np.reshape((data.astype(dtype="float32") / 255.0), [1, 1080, 1920, 3])
    img.close()
    return data

def postprocess_pil_image(npdata):
    """ Postprocess output data from our keras model into a PIL image """
    npdata = np.asarray(np.clip(npdata * 255, 0, 255), dtype="uint8")
    return PIL.Image.fromarray(npdata, "RGB")

if __name__ == '__main__':
    main()
