import tensorflow as tf
from fractal_gen import FractalGenModel
import os

snapshot_path = '.data/snapshots'
cache_path = '.data/fractal_cache'

if __name__ == '__main__':
    gen = FractalGenModel()
    for img in gen.generate_sequence():
        break