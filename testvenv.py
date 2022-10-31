import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pathlib 

import tqdm
import numpy as np
import cv2
import albumentations

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from model import deeplabv3_plus
from dataprocessing import load_data
from metrics import dice_coef, iou, dice_coef, dice_loss
from train import tf_dataset, shuffling

print('Imported venv libraries successfully')

W, H = 512, 512

if __name__ == "__main__":
    """ Seeding the environement """

    np.random.seed(42)
    tf.random.set_seed(42)

    ''' Directory for storing the training output '''
    save_training_path = pathlib.Path('./trained_model')
    save_training_path.mkdir(parents=True, exist_ok=True)

    """ Hyperparameters """
    batch_size = 2
    learning_rate = 1e-4
    epochs_to_train = 20
    model_path = pathlib.Path(os.path.join(save_training_path, "model-cedar.h5"))
    csv_path = pathlib.Path(os.path.join(save_training_path, "data.csv"))

    """ Dataset """
    dataset_path = pathlib.Path('./model_data')
    training_path = pathlib.Path(os.path.join(dataset_path, 'train'))
    testing_path = pathlib.Path(os.path.join(dataset_path, 'test'))

    print("LOAD THE TRAINING IMAGES AND MASKS")
    train_images_collection, train_masks_collection = load_data(training_path, do_splitting=False, ext1='*.png', as_str=True)
    print("SHUFFLE THE TRAINING IMAGES AND MASKS")
    train_images_collection, train_masks_collection = shuffling(train_images_collection, train_masks_collection)
    print("LOAD THE TESTING IMAGES AND MASKS")
    test_images_collection, test_masks_collection = load_data(testing_path, do_splitting=False, ext1='*.png', as_str=True)

    print(f"Train <#images, #masks>: <{len(train_images_collection)}, {len(train_masks_collection)}>")
    print(f"Test <#images, #masks>: <{len(test_images_collection)}, {len(test_masks_collection)}>")

    print('CREATE THE TRAINING DATASET TENSORS')
    train_dataset = tf_dataset(train_images_collection, train_masks_collection, batch = batch_size)
    print('CREATE THE TESTING DATASET TENSORS')
    test_dataset = tf_dataset(test_images_collection, test_masks_collection, batch = batch_size)

    print('INITIALIZE THE DEEPLABV3 MODEL')
    model = deeplabv3_plus((H, W, 3))

    print('COMPILE THE DEEPLAB V3 MODEL')
    model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    print('PRELOAD WEIGHTS')
    model.load_weights(model_path)
    print('REEVALUATE THE MODEL')
    # Re-evaluate the model
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

