import os
import datetime
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import PIL

from train_data import TrainData
from fractal_gen import FractalGenTensorflowModel
from fractal_model import build_model

logging.basicConfig(
    format='[%(asctime)s][%(levelname)-5.5s] %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('.data/log.txt'),
        logging.StreamHandler()
    ])

# constants, directory paths
tensorboard_dir = '.data\\tensorboard' # this uses backslash because the internal keras tensorboard callback improperly handles forward slash paths
train_checkpoints_dir = '.data/model_checkpoints'
sample_inputs_dir = '.data/model_sample_inputs'
sample_outputs_dir = '.data/model_samples'

# private vars, used to contain state that can be referenced by keras events
# initialized in __main__
_current_epoch = 1
_model = None

class TrainProcessor():
    def __init__(self):
        self._current_epoch = 1
        self._model = None

        def get_last_checkpoint():
            ld = [int(x) for x in os.listdir(train_checkpoints_dir)]
            ld.sort()

            if len(ld) > 0:
                return int(ld[-1])

        self._model = build_model()
        last_checkpoint = get_last_checkpoint()
        if last_checkpoint:
            self._current_epoch = last_checkpoint + 1
            self._model.load_weights(f'{train_checkpoints_dir}/{last_checkpoint}/model_weights')

    def train(self, epochs, steps_per_epoch=12, batch_size=4):
        all_train_callbacks = [
            # tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),

            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch_num, logs: self._on_train_epoch_begin(),
                on_epoch_end=lambda epoch_num, logs: self._on_train_epoch_end(logs['loss'])
            )
        ]

        batch_gen = self._train_generator(batch_size=batch_size)
        batch_iter = 0
        
        for features, labels in batch_gen:
            self._model.fit(
                x=features,
                y=labels,
                epochs=1,
                shuffle=False,
                callbacks=all_train_callbacks)

            batch_iter += 1
            if batch_iter >= epochs: break

    def preprocess_pil_image(img):
        """ Preprocess PIL image into the input data type for our keras model """
        data = np.asarray(img, dtype=np.uint8)
        data = np.reshape((data.astype(dtype=np.float32) / 255.0), [1, 1080, 1920, 3])
        img.close()
        return data

    def postprocess_pil_image(npdata):
        """ Postprocess output data from our keras model into a PIL image """
        npdata = np.asarray(np.clip(npdata[0] * 255, 0, 255), dtype=np.uint8)
        return PIL.Image.fromarray(npdata, 'RGB')

    def _on_train_epoch_begin(self):
        logging.info(f'epoch begin {self._current_epoch}')

    def _on_train_epoch_end(self, loss):
        epoch = self._current_epoch

        logging.info(f'epoch end {epoch}, loss={loss}')

        if (epoch % 10) == 0:
            self._save_model_checkpoint()
            self._process_sample_images()

        self._current_epoch += 1
        
    def _train_generator(self, batch_size=8):
        features = []
        labels = []

        while True:
            td = TrainData.get_random()
            feature = TrainProcessor.preprocess_pil_image(td.get_train_image())
            label = TrainProcessor.preprocess_pil_image(td.get_next_train_image())

            features.append(feature[0])
            labels.append(label[0])

            if len(features) >= batch_size:
                yield (np.array(features), np.array(labels))

                features = []
                labels = []

    def _save_model_checkpoint(self):
        model = self._model
        epoch = self._current_epoch

        logging.info(f'begin save model weights after epoch {epoch}')

        out_dir = f'{train_checkpoints_dir}/{epoch}'
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        model.save_weights(f'{out_dir}/model_weights')

        logging.info(f'model weights after epoch {epoch} save end')

    def _process_sample_images(self):
        """ Processes images in the '.data/model_sample_inputs' directory through the model, each with 5 samples """

        model = self._model
        epoch = self._current_epoch

        for img in os.listdir(sample_inputs_dir):
            out_dir = f'{sample_outputs_dir}/{img}'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            logging.info(f'process sample {img}')

            try:
                x = PIL.Image.open(f'{sample_inputs_dir}/{img}')
                x.load()
                x.save(f'{out_dir}/{epoch}-0.png')

                x = TrainProcessor.preprocess_pil_image(x)

                for i in range(1, 6):
                    x = model.predict(x)

                    y = TrainProcessor.postprocess_pil_image(x)
                    y.save(f'{out_dir}/{epoch}-{i}.png')
                    y.close()

                    logging.info(f'process sample {img} completed {i}/5')
            except Exception as e:
                logging.error(f'exception processing sample {img}', exc_info=True)
                pass

if __name__ == '__main__':
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

    proc = TrainProcessor()
    proc.train(10000)
