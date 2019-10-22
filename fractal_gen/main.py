import tensorflow as tf
from fractal_gen import FractalGenModel
import os
from moviepy.editor import ImageSequenceClip, ImageClip, concatenate


def gif():
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    clip = ImageSequenceClip('.data/imgs',fps=24)
    clip.write_videofile('movie.avi',audio=False,codec='png')

if __name__ == '__main__':
    gen = FractalGenModel()
    for img in gen.generate_sequence():
        break