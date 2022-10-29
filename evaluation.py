import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pathlib

import numpy as np
import cv2
import sklearn

import tensorflow as tf

from model import deeplabv3_plus

W, H = 512, 512



if __name__ == "__main__":
    """ Seeding the environment """
    np.random.seed(42)
    tf.random.set_seed(42)

    results = pathlib.Path("./files/")
    results.mkdir(parents=True, exist_ok=True)

    """ Load the trained model """

    print('INITIALIZE THE DEEPLABV3 MODEL')
    model = deeplabv3_plus((H, W, 3))

    model.load_weights(f"{results.resolve()}")
    