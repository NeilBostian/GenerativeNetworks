import os
import random
import pickle
import datetime

import torch
import numpy as np
from PIL import Image

from model import Model
from train_data import TrainData
from loss_train_data import get_loss_train_data

class ModelProcessor():
    def __init__(self, path):
        self.path = path
        self.device = torch.device('cuda')
        self.model = Model(self.device)
        self._load_model()

    def train_frames(self):
        if not self._loss_trained:
            raise os.error('Loss has not been trained yet (call ModelProcessor.train_loss())')

        for x, y, _ in ModelProcessor._train_frames_iter(300000, 1):
            epoch = self._epoch

            loss = self.model.train_frame(x, y)
            print(f'{datetime.datetime.now()} train_frame epoch {epoch} loss={loss}')

            self._epoch = self._epoch + 1

            if (epoch % 500) == 0:
                self._checkpoints[epoch] = {
                    'epoch': epoch,
                    'loss': loss
                }

                self.model.save(self._path(f'ckpt-{epoch}.pt'))
                self._save_model()

                self._process_sample_images()

    def train_loss(self):
        if self._loss_trained:
            raise os.error('Loss has already been trained on this model')

        for x, y, epoch in ModelProcessor._train_loss_iter(400, 4):
            loss = self.model.train_loss(x, y)
            print(f'{datetime.datetime.now()} train_loss epoch {epoch} loss={loss}')

        self._loss_trained = True
        self._checkpoints[1] = {
            'epoch': 1,
            'loss': None
        }

        self.model.save(self._path(f'ckpt-1.pt'))
        self._save_model()

    def _load_model(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if not os.path.exists(self._path('index')):
            self._loss_trained = False
            self._epoch = 1
            self._checkpoints = { }
            self._save_model()
        else:
            with open(self._path('index'), 'rb') as f:
                mdata = pickle.load(f)
                self._loss_trained = mdata['loss_trained']
                self._epoch = mdata['epoch']
                self._checkpoints = mdata['checkpoints']
            
            if len(self._checkpoints) > 0:
                latest_checkpoint = max(self._checkpoints)
                ckpt_path = self._path(f'ckpt-{latest_checkpoint}.pt')
                if os.path.exists(ckpt_path):
                    self.model.load(ckpt_path)
                else:
                    self.model.load(self._path('ckpt-1.pt'))
                    self._epoch = 1
                    self._checkpoints = {
                        1: {'epoch': 1, 'loss': None}
                    }

    def _save_model(self):
        with open(self._path('index'), 'wb') as f:
            mdata = {
                'loss_trained': self._loss_trained,
                'epoch': self._epoch,
                'checkpoints': self._checkpoints
            }
            pickle.dump(mdata, f)

    def _path(self, *paths):
        return os.path.join(self.path, *paths)

    def _train_frames_iter(num_batches, batch_size):
        def _train_frames_iter_singles():
            for i in range(0, batch_size * num_batches):
                td = TrainData.get_random()
                x = td.get_train_image()
                y = td.get_next_train_image()

                yield (x, y, i)

        xs = []
        ys = []

        for x, y, i in _train_frames_iter_singles():
            xs.append(x[0])
            ys.append(y[0])

            if len(xs) >= batch_size:
                epoch = int((i + 1) / batch_size)
                yield (np.array(xs), np.array(ys), epoch)

                xs = []
                ys = []

    def _train_loss_iter(num_batches, batch_size):
        def _train_loss_iter_singles():
            for i in range(0, batch_size * num_batches):
                g = random.randint(0, 1)
                if g == 0:
                    x = TrainData.get_random().get_train_image()
                    y = 0
                else:
                    x = get_loss_train_data()
                    y = 1

                yield (x, y, i)

        xs = []
        ys = []

        for x, y, i in _train_loss_iter_singles():
            xs.append(x[0])
            ys.append(y)

            if len(xs) >= batch_size:
                epoch = int((i + 1) / batch_size)
                yield (np.array(xs), np.array(ys), epoch)

                xs = []
                ys = []

    def _process_sample_images(self):
        """ Processes images in the '.data/model_sample_inputs' directory through the model, each with 5 samples """

        model = self.model
        epoch = self._epoch

        for img in os.listdir('.data/model_sample_inputs'):
            sample_outputs = self._path('sample_outputs')
            if not os.path.exists(sample_outputs):
                os.mkdir(sample_outputs)

            out_dir = self._path(f'sample_outputs', img)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            print(f'process sample {img}')

            try:
                x = Image.open(f'.data/model_sample_inputs/{img}')
                x.load()
                x.save(f'{out_dir}/{epoch}-0.png')

                x = TrainData.preprocess_pil_image(x)
                max_iters = 4
                for i in range(1, max_iters + 1):
                    x = model.get_frame(x)

                    y = TrainData.postprocess_pil_image(x)
                    y.save(f'{out_dir}/{epoch}-{i}.png')
                    y.close()

                    print(f'process sample {img} completed {i}/{max_iters}')
            except Exception as e:
                print(f'exception processing sample {img} {e}')
                pass