import os
import pathlib
from tqdm import tqdm

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.core.serialization import SerializableMeta as ASerializableMeta

def load_data(path:pathlib.Path, split: float=0.1, do_splitting: bool=True, ext1: str='*.jpg', ext2: str='*.png', as_str:bool = False) -> tuple:
    """
        path: pathlib.Path - The directory path to the dataset
        split: float - How much of the dataset to allocate for testing purposes. 
                       Value of 0.1 implies 10% of data reserved for testing
    """

    """ GETTING THE LIST OF IMAGES AND MASKS IN THE DATASET FOLDER"""

    image_files = sorted(pathlib.Path(os.path.join(path, "images")).glob(ext1))
    mask_files = sorted(pathlib.Path(os.path.join(path, "masks")).glob(ext2))

    if(as_str):
        image_files = [f"{i_path.resolve()}" for i_path in image_files]
        mask_files = [f"{i_path.resolve()}" for i_path in mask_files]
    else:
        image_files = [i_path.resolve() for i_path in image_files]
        mask_files = [i_path.resolve() for i_path in mask_files]
    
    if(do_splitting):
        test_split_size = int(len(image_files) * split)

        train_images, test_images = train_test_split(image_files, test_size=test_split_size, random_state=42)
        train_masks, test_masks = train_test_split(mask_files, test_size=test_split_size, random_state=42)

        return train_images, test_images, train_masks, test_masks
    
    return image_files, mask_files

def getTransformed(effect: ASerializableMeta, cv_image:np.ndarray, cv_mask:np.ndarray, **kwargs) -> tuple:
        transform = effect(**kwargs)
        transform_result = transform(image=cv_image, mask=cv_mask)
        return transform_result['image'], transform_result['mask']

def augment_data(images_path_collection: list, masks_path_collection: list, save_to_directory: pathlib.Path, augment=True) -> None:
    
    H, W = 512, 512
    total_images = len(images_path_collection)
    save_image_path = pathlib.Path.joinpath(save_to_directory, "images")
    save_masks_path = pathlib.Path.joinpath(save_to_directory, "masks")

    save_image_path.mkdir(parents=True, exist_ok=True)
    save_masks_path.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in tqdm(zip(images_path_collection, masks_path_collection), total=total_images):

        cv_image = cv2.imread(f"{image_path.resolve()}", cv2.IMREAD_COLOR)
        cv_mask = cv2.imread(f"{mask_path.resolve()}", cv2.IMREAD_COLOR)

        
        """ Augmentation is true then apply augmentation"""
        if(augment):
            """ Apply Center cropping """

            image1, mask1 = getTransformed(A.PadIfNeeded, cv_image, cv_mask, min_width=W, min_height=H, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, position=A.PadIfNeeded.PositionType.CENTER)
            image2, mask2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
            image3, mask3 = getTransformed(A.ChannelShuffle, image1, mask1, p=1.0)
            image4, mask4 = getTransformed(A.CoarseDropout, image1, mask1, p=1.0, min_holes=3, max_holes=10, max_height=32, max_width=32)
            image5, mask5 = getTransformed(A.Rotate, image1, mask1, limit=45, p=1.0)
            
            images = [image1, image2, image3, image4, image5]
            masks = [mask1, mask2, mask3, mask4, mask5]

        else:
            image1, mask1 = getTransformed(A.PadIfNeeded, cv_image, cv_mask, min_width=W, min_height=H, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, position=A.PadIfNeeded.PositionType.CENTER)
            """ We need atleast the padding for consistent 512 x 512 image size (based on W X H values at the top of this method)"""
            images = [image1]
            masks = [mask1]
        
        for index, (im, msk) in enumerate(zip(images, masks)):
            try:
                """ Apply Center cropping """
                transform = A.Compose([
                            A.CenterCrop(height=H, width=W, p=1.0)
                        ],# additional_targets={"mask": "mask"}
                    )
                transform_result = transform(image=im, mask=msk)
                cropped_image = transform_result['image']
                cropped_mask = transform_result['mask']
            except:
                cropped_image = cv2.resize(im, (W, H))
                cropped_mask = cv2.resize(msk, (W, H))

            
            image_save_path = pathlib.Path.joinpath(save_image_path, f"{image_path.stem}-{index}.png")
            mask_save_path = pathlib.Path.joinpath(save_masks_path, f"{mask_path.stem}-{index}.png")
            cv2.imwrite(f"{image_save_path}", cropped_image)
            cv2.imwrite(f"{mask_save_path}", cropped_mask)


if __name__ == '__main__':
    """ Path to the dataset """
    data_path = pathlib.Path("./people_segmentation")

    """ Path to save the training images """
    training_images_path = pathlib.Path("./model_data/train/images")
    training_images_path.mkdir(parents=True, exist_ok=True)
    
    """ Path to save the training masks """
    training_masks_path = pathlib.Path("./model_data/train/masks")
    training_masks_path.mkdir(parents=True, exist_ok=True)

    """ Path to save the testing images """
    testing_images_path = pathlib.Path("./model_data/test/images")
    testing_images_path.mkdir(parents=True, exist_ok=True)

    """ Path to save the testing masks """
    testing_masks_path = pathlib.Path("./model_data/test/masks")
    testing_masks_path.mkdir(parents=True, exist_ok=True)



    """ Seeding the environement """
    np.random.seed(42)

    """ Loading the dataset """    
    train_images, test_images, train_masks, test_masks = load_data(data_path, split=0.2)

    print(f"Train <#images, #masks>: <{len(train_images)}, {len(train_masks)}>")
    print(f"Test <#images, #masks>: <{len(test_images)}, {len(test_masks)}>")

    # """ Data Augmentation """
    augment_data(train_images, train_masks, training_images_path.parents[0], augment=True)
    augment_data(test_images, test_masks, testing_images_path.parents[0], augment=False)

    # test_img = cv2.imread('./people_segmentation/images/001.jpg', cv2.IMREAD_COLOR)
    # test_mask = cv2.imread('./people_segmentation/masks/001.png', cv2.IMREAD_COLOR)
    # t_image, t_mask = getTransformed(A.PadIfNeeded, test_img, test_mask, min_width=512, min_height=512, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0, position=A.PadIfNeeded.PositionType.CENTER, mask_value=0)
    # cv2.imwrite('./t_image.png', t_image)
    # cv2.imwrite('./t_mask.png', t_mask*255)
    # print(test_img.shape, test_mask.shape)
