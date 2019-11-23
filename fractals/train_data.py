import os
import logging
import random
import tensorflow as tf
import numpy as np
from PIL import Image

from fractal_gen import FractalGenTensorflowModel

THETA_BOUNDS = 600

if not os.path.exists('.data'):
    os.mkdir('.data')

if not os.path.exists('.data/train_img_cache'):
    os.mkdir('.data/train_img_cache')

ratio = 9.0 / 16.0
start_x = -2.3
end_x = start_x * -1.
start_y = start_x * ratio
end_y = end_x * ratio
width = 1920 # image width
step = (end_x - start_x) / width
Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
Z = X + 1j * Y
        
tfmodel = FractalGenTensorflowModel(Z.shape, Z.dtype)

class TrainData():
    _all_bg_ratios = [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
        (3.0, 2.0, 1.0),
        (4.0, 2.5, 1.0),
        (4.5, 2.0, 1.0)
    ]

    _all_bg_shuffles = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ]

    _all_cs = [
        lambda theta: -(0.835 - 0.05 * np.cos(theta)) - (0.2321 + 0.05 * np.sin(theta)) * 1j,
        lambda theta: -(0.835 - 0.03 * np.cos(theta)) - (0.2321 + 0.08 * np.sin(theta)) * 1j,
        lambda theta: -(0.835 - 0.01 * np.cos(theta)) - (0.2321 + 0.11 * np.sin(theta)) * 1j,
        lambda theta: -(0.835 - 0.05 * np.sin(theta)) - (0.2321 + 0.05 * np.cos(theta)) * 1j,
        lambda theta: -(0.835 - 0.03 * np.sin(theta)) - (0.2321 + 0.08 * np.cos(theta)) * 1j,
        lambda theta: -(0.835 - 0.01 * np.sin(theta)) - (0.2321 + 0.11 * np.cos(theta)) * 1j,
        lambda theta: -(0.805 - 0.05 * np.cos(theta)) - (0.2621 + 0.05 * np.sin(theta)) * 1j,
        lambda theta: -(0.805 - 0.03 * np.cos(theta)) - (0.2621 + 0.08 * np.sin(theta)) * 1j,
        lambda theta: -(0.805 - 0.01 * np.cos(theta)) - (0.2621 + 0.11 * np.sin(theta)) * 1j,
        lambda theta: -(0.805 - 0.05 * np.sin(theta)) - (0.2621 + 0.05 * np.cos(theta)) * 1j,
        lambda theta: -(0.805 - 0.03 * np.sin(theta)) - (0.2621 + 0.08 * np.cos(theta)) * 1j,
        lambda theta: -(0.805 - 0.01 * np.sin(theta)) - (0.2621 + 0.11 * np.cos(theta)) * 1j
    ]

    def __init__(self, theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind):
        self._theta_iter = theta_iter
        self._bg_ratio_ind = bg_ratio_ind
        self._bg_ratio_shuffle = bg_ratio_shuffle
        self._c_ind = c_ind

    def _cache_train_image(self, theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind):
        bg_ratio = tuple([self._all_bg_ratios[bg_ratio_ind][x] for x in self._all_bg_shuffles[bg_ratio_shuffle]])
        
        theta = 2 * np.pi * theta_iter / THETA_BOUNDS
        c = self._all_cs[c_ind](theta)

        result_img = tfmodel.generate_image(Z, c, bg_ratio, (0.9, 0.9, 0.9))

        result_img.save(self._get_image_path(theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind))

        return result_img

    def _get_image_path(self, theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind):
        return f'.data/train_img_cache/{bg_ratio_ind}-{bg_ratio_shuffle}-{c_ind}-{theta_iter}.png'

    def get_train_image(self):
        fname = self._get_image_path(self._theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)

        if not os.path.exists(fname):
            logging.info(f'image does not exist, creating from fractal_gen: {fname}')
            img = self._cache_train_image(self._theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)
            return img
        else:
            logging.info(f'image exists, using cached version: {fname}')
            img = Image.open(fname)
            img.load()
            return img

    def get_next_train_image(self):
        next_theta_iter = (self._theta_iter + 1)  % THETA_BOUNDS
        fname = self._get_image_path(next_theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)

        if not os.path.exists(fname):
            logging.info(f'image does not exist, creating from fractal_gen: {fname}')
            img = self._cache_train_image(next_theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)
            return img
        else:
            logging.info(f'image exists, using cached version: {fname}')
            img = Image.open(fname)
            img.load()
            return img

    def get_random():
        theta_iter = random.randint(0, THETA_BOUNDS - 1)
        bg_ratio_ind = random.randint(0, len(TrainData._all_bg_ratios) - 1)
        bg_ratio_shuffle = random.randint(0, len(TrainData._all_bg_shuffles) - 1)
        c_ind = random.randint(0, len(TrainData._all_cs) - 1)

        return TrainData(theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind)
        