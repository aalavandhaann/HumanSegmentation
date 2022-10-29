import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pathlib

import numpy as np
import cv2
import sklearn

import tensorflow as tf

W, H = 512, 512



if __name__ == "__main__":
    """ Seeding the environment """
    np.random.seed(42)
    tf.random.set_seed(42)

    results = pathlib.Path("./predictions/")
    results.mkdir(parents=True, exist_ok=True)

    """ Load the trained model """
    with tf.keras.utils.Cu