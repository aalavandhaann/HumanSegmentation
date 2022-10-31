import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pathlib

import numpy as np
import cv2
import sklearn
print('LOADED NUMPY, CV2, SKLEARN')
print('NOW ATTEMPTING TO LOAD TENSORFLOW')
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print('FINISHED LOADING TENSORFLOW MODULE')

from model import deeplabv3_plus
from dataprocessing import load_data
from metrics import dice_coef, iou, dice_coef, dice_loss


W, H = 512, 512

def shuffling(x: list, y: list) -> tuple:
    x, y = sklearn.utils.shuffle(x, y, random_state=42)
    return x, y

def read_image(path: pathlib.Path) -> np.ndarray:
    return (cv2.imread(f"{path.decode()}", cv2.IMREAD_COLOR) / 255.0).astype(np.float32)

def read_mask(path:pathlib.Path) -> np.ndarray:
    mask = (cv2.imread(f"{path.decode()}", cv2.IMREAD_GRAYSCALE)).astype(np.float32)
    return np.expand_dims(mask, axis=-1)

def tf_parse(image_path: pathlib.Path, mask_path: pathlib.Path) -> tuple:
    def __parse__(im_p: pathlib.Path, msk_p: pathlib.Path) -> tuple:
        return read_image(im_p), read_mask(msk_p)
    
    image, mask = tf.numpy_function(__parse__, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 1])

    return image, mask

def tf_dataset(images_collection: list, masks_collection: list, batch: int = 2):
    dataset = tf.data.Dataset.from_tensor_slices((images_collection, masks_collection))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding the environement """

    np.random.seed(42)
    tf.random.set_seed(42)

    ''' Directory for storing the training output '''
    save_training_path = pathlib.Path('./trained_model')
    save_training_path.mkdir(parents=True, exist_ok=True)


    """ Hyperparameters """
    batch_size = 8
    learning_rate = 1e-4
    epochs_to_train = 20
    model_path = pathlib.Path(os.path.join(save_training_path, "model.h5"))
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
    model.compile(loss=dice_loss, run_eagerly=True, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    
    model_path = str(model_path.resolve())
    csv_path = str(csv_path.resolve())
    
    print('PATH WHERE THE MODEL IS SAVED : ', model_path)
    print('PATH WHERE THE CSV IS SAVED : ', csv_path)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_weights_only=True, save_freq='epoch'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(csv_path),
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
    ]

    print('REGISTERED ALL THE CALLBACKS ')

    print('ALL DONE GO AHEAD AND FIT THE MODEL')
    print('TRAINING STARTS ......')
    model.fit(train_dataset, epochs=epochs_to_train, validation_data=test_dataset, callbacks=callbacks)
    model.save(model_path)
    print('TRAINING ENDED......')