import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pathlib

from tqdm import tqdm
import numpy as np
import cv2
import sklearn

import tensorflow as tf

from dataprocessing import load_data
from model import deeplabv3_plus
from metrics import dice_coef, iou, dice_coef, dice_loss

W, H = 512, 512

def save_prediction(original_image, ground_truth_mask, predicted_mask, save_as_path):
    """ A GRAY LINE TO SEPARATE BETWEEN THE RESULTS """
    separator_line = np.ones((H, 10, 3)) * 128

    """ The ground_truth_mask is a grayscale image with shape H x W and each cell refers to a scalar value """
    """ Hence we need to expand the dimension by 1 to hold the integer value for either 0 or 255 """
    """ So we need to add an additional dimension such that the shape becomes H x W x 1"""
    """ 
        Infact we can also use H x W without expading the dimensions but to keep an uniform shape that conforms
        to a tensor which always is H x W x (number of channels), we need to expand the dimension by the necessary 
        number of channels. For example opencv when imread a color image will give you H x W x 3; when imread a 
        RGBA image will give you H x W x 4; with a grayscale image it doesn't have an extra dimension apart from 
        the height(H) and width (W), why? Because you don't need an additional list in each pixel of the image because 
        they are going to hold only a single value from 0 to 255 (from black to white)
        But a tensor even if it is a grayscale image will always be H x W x 1 instead of just H x W
    
    """
    mask = np.expand_dims(ground_truth_mask*255, axis = -1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    """ The predicted_mask is always a single channel grayscale, so need to convert to 3 channel """
    prediction = np.expand_dims(predicted_mask*255, axis = -1)
    prediction = np.concatenate([prediction, prediction, prediction], axis=-1)

    image_as_layout = np.concatenate([original_image, separator_line, mask, separator_line, prediction], axis=1)
    cv2.imwrite(f"{save_as_path.resolve()}", image_as_layout)


if __name__ == "__main__":
    """ Seeding the environment """
    np.random.seed(42)
    tf.random.set_seed(42)

    saved_weights = pathlib.Path("./trained_model/model-graham.h5")
    results = pathlib.Path("./trained_model/model.h5")
    results.mkdir(parents=True, exist_ok=True)

    """ Dataset """
    dataset_path = pathlib.Path('./model_data')
    training_path = pathlib.Path(os.path.join(dataset_path, 'train'))
    testing_path = pathlib.Path(os.path.join(dataset_path, 'test'))

    """ Load the trained model """

    print('INITIALIZE THE DEEPLABV3 MODEL')
    model = deeplabv3_plus((H, W, 3))

    with tf.keras.utils.CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model.load_weights(f"{saved_weights.resolve()}")
    
    """ Load the test dataset """
    print("LOAD THE TESTING IMAGES AND MASKS")
    test_images_collection, test_masks_collection = load_data(testing_path, do_splitting=False, ext1='*.png', as_str=False)
    print(f"Test <#images, #masks>: <{len(test_images_collection)}, {len(test_masks_collection)}>")



    """ Prediction of model for segmentation of humans """

    predictions_save_dir = pathlib.Path('./predictions/')
    predictions_save_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in tqdm(zip(test_images_collection, test_masks_collection), total=len(test_images_collection)):
        """ LOADING THE IMAGE IN OPENCV"""
        image = cv2.imread(f"{image_path}", cv2.IMREAD_COLOR)
        image_as_tensor = image / 255.0
        image_as_tensor = np.expand_dims(image_as_tensor, axis=0)

        """ LOADING THE GROUND TRUTH MASK IN OPENCV"""
        mask_ground_truth = cv2.imread(f"{mask_path}", cv2.IMREAD_GRAYSCALE)

        """ MASK PREDICTED BY THE MODEL"""
        mask_predicted = model.predict(image_as_tensor)[0]
        mask_predicted = np.squeeze(mask_predicted, axis=-1)

        mask_predicted = mask_predicted > 0.5
        mask_predicted = mask_predicted.astype(np.int32)

        """ Let us analyze the image vs ground truth mask vs predicted mask """
        save_image_path = pathlib.Path(os.path.join(predictions_save_dir, '%s.png'%(image_path.stem)))

        save_prediction(image, mask_ground_truth, mask_predicted, save_image_path)
        
        # break