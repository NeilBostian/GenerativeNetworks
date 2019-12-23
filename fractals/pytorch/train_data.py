import os
import logging
import random
import numpy as np
from PIL import Image

THETA_BOUNDS = 600

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
        lambda theta: -(0.835 - 0.05 * np.cos(theta)) - (0.2321 + 0.05 * np.sin(theta)) * 1j
        # lambda theta: -(0.835 - 0.03 * np.cos(theta)) - (0.2321 + 0.08 * np.sin(theta)) * 1j,
        # lambda theta: -(0.835 - 0.01 * np.cos(theta)) - (0.2321 + 0.11 * np.sin(theta)) * 1j,
        # lambda theta: -(0.835 - 0.05 * np.sin(theta)) - (0.2321 + 0.05 * np.cos(theta)) * 1j,
        # lambda theta: -(0.835 - 0.03 * np.sin(theta)) - (0.2321 + 0.08 * np.cos(theta)) * 1j,
        # lambda theta: -(0.835 - 0.01 * np.sin(theta)) - (0.2321 + 0.11 * np.cos(theta)) * 1j,
        # lambda theta: -(0.805 - 0.05 * np.cos(theta)) - (0.2621 + 0.05 * np.sin(theta)) * 1j,
        # lambda theta: -(0.805 - 0.03 * np.cos(theta)) - (0.2621 + 0.08 * np.sin(theta)) * 1j,
        # lambda theta: -(0.805 - 0.01 * np.cos(theta)) - (0.2621 + 0.11 * np.sin(theta)) * 1j,
        # lambda theta: -(0.805 - 0.05 * np.sin(theta)) - (0.2621 + 0.05 * np.cos(theta)) * 1j,
        # lambda theta: -(0.805 - 0.03 * np.sin(theta)) - (0.2621 + 0.08 * np.cos(theta)) * 1j,
        # lambda theta: -(0.805 - 0.01 * np.sin(theta)) - (0.2621 + 0.11 * np.cos(theta)) * 1j
    ]

    def __init__(self, theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind, logging=True):
        self._theta_iter = theta_iter
        self._bg_ratio_ind = bg_ratio_ind
        self._bg_ratio_shuffle = bg_ratio_shuffle
        self._c_ind = c_ind
        self._logging = logging

    def _get_image_path(self, theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind):
        return f'.data/train_img_cache/{bg_ratio_ind}-{bg_ratio_shuffle}-{c_ind}-{theta_iter}.png'

    def get_train_image(self):
        fname = self._get_image_path(self._theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)

        img = Image.open(fname)
        img.load()
        return TrainData.preprocess_pil_image(img)

    def get_next_train_image(self):
        next_theta_iter = (self._theta_iter + 1)  % THETA_BOUNDS
        fname = self._get_image_path(next_theta_iter, self._bg_ratio_ind, self._bg_ratio_shuffle, self._c_ind)

        img = Image.open(fname)
        img.load()
        return TrainData.preprocess_pil_image(img)

    def preprocess_pil_image(img):
        """ Preprocess PIL image into the input data type for our keras model """
        data = np.asarray(img, dtype=np.uint8)
        data = np.transpose(np.reshape((data.astype(dtype=np.float32) / 255.0), [1, 1080, 1920, 3]), (0, 3, 1, 2))
        img.close()
        return data

    def postprocess_pil_image(npdata):
        """ Postprocess output data from our keras model into a PIL image """
        npdata = np.asarray(np.clip(np.transpose(npdata[0], (1, 2, 0)) * 255, 0, 255), dtype=np.uint8)
        return Image.fromarray(npdata, 'RGB')

    def get_random(logging=True):
        theta_iter = random.randint(0, int((THETA_BOUNDS - 1) / 2))
        bg_ratio_ind = random.randint(0, len(TrainData._all_bg_ratios) - 1)
        bg_ratio_shuffle = random.randint(0, len(TrainData._all_bg_shuffles) - 1)
        c_ind = random.randint(0, len(TrainData._all_cs) - 1)

        return TrainData(theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind, logging)

    def get_all():
        for theta_iter in range(0, THETA_BOUNDS):
            for bg_ratio_ind in range(0, len(TrainData._all_bg_ratios)):
                for bg_ratio_shuffle in range(0, len(TrainData._all_bg_shuffles)):
                    for c_ind in range(0, len(TrainData._all_cs)):
                        yield TrainData(theta_iter, bg_ratio_ind, bg_ratio_shuffle, c_ind)
