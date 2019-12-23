import os
import random

import numpy as np
from PIL import Image

def get_loss_train_data():
    if not os.path.exists('.data/DIV2K'):
        # DIV2K Home Page: https://data.vision.ee.ethz.ch/cvl/DIV2K/
        # DIV2K Training Set: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
        raise os.error('No DIV2K Training set found in .data/DIV2K directory. Download http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip')

    def format_loss_train_input_image(img):
        """ Crops any image larger than 1920x1080 and formats the pixels to numpy array of shape [batches, channels, height, width] """
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

        return np.transpose(np.reshape(data, [1, 1080, 1920, 3]), (0, 3, 1, 2))

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
    return format_loss_train_input_image(x)