import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras


def SqueezeAndExcite(inputs: keras.engine.keras_tensor.KerasTensor, ratio: int=8) -> keras.engine.keras_tensor.KerasTensor:
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = init * se

    return x

"""
    ASPP - Atrous spatial pyramid pooling
    A semantic segmentation module for resampling a given
    feature layer at multiple rates prior to convolution. 
    More Info @ https://paperswithcode.com/method/aspp
    This is the method that does the encoder part as shown in the image
    https://miro.medium.com/max/1037/1*2mYfKnsX1IqCCSItxpXSGA.png 
    or see the image "deeplabv3-architechture.png" in this folder 
"""
def ASPP(inputs: keras.engine.keras_tensor.KerasTensor) -> keras.engine.keras_tensor.KerasTensor:
    print(type(inputs))
    shape = inputs.shape       
    # print('SHAPE OF THE INPUT :: ', shape) 
    """ 1 x 1 convolution the first layer in the encoder (see image 'deeplabv3-architechture.png') """
    y2 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.Activation('relu')(y2)
    # print("FIRST LAYER SHAPE :: ", y2.shape)

    """ 3 x 3 convolution dilation rate = 6 the second layer in the encoder (see image 'deeplabv3-architechture.png') """
    y3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=6)(inputs)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.Activation('relu')(y3)
    # print("SECOND LAYER SHAPE :: ", y3.shape)

    """ 3 x 3 convolution dilation rate = 12 the third layer in the encoder (see image 'deeplabv3-architechture.png')  """
    y4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=12)(inputs)
    y4 = tf.keras.layers.BatchNormalization()(y4)
    y4 = tf.keras.layers.Activation('relu')(y4)
    # print("THIRD LAYER SHAPE :: ", y4.shape)

    """ 3 x 3 convolution dilation rate = 18 the fourth layer in the encoder (see image 'deeplabv3-architechture.png')  """
    y5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=18)(inputs)
    y5 = tf.keras.layers.BatchNormalization()(y5)
    y5 = tf.keras.layers.Activation('relu')(y5)
    # print("FOURTH LAYER SHAPE :: ", y5.shape)    

    """ Image pooling the last layer in the encoder (see image 'deeplabv3-architechture.png') """

    ''' The below step converts the 32 x 32 (or based on the inputs.shape) tensor to 1 x 1 '''
    y1 = tf.keras.layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)    
    ''' Until the tensor is upsample the below steps operate on the 1 x 1 tensor as input'''
    y1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y1)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.keras.layers.Activation('relu')(y1)    
    ''' Now Upsample the 1 x 1 back to 32 x 32 (or based on the inputs.shape)'''
    y1 = tf.keras.layers.UpSampling2D((shape[1], shape[2]), interpolation= 'bilinear') (y1)
    # print("FIFTH(LAST) LAYER SHAPE :: ", y1.shape)

    ''' Combine all the layers to one single layer '''
    y = tf.keras.layers.Concatenate()([y1, y2, y3, y4, y5])
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)    
    # print("COMBINED ALL LAYERS SHAPE :: ", y.shape)

    return y


def deeplabv3_plus(shape: tuple) -> tf.keras.models.Model:
    """ Input for the given shape """
    inputs = tf.keras.Input(shape)

    """ 
        Initialize the encoder (using resnet50 as opposed to DCNN in the image "deeplabv3-architechture.png")
        Keyword Arguments: 
        "weights": Use the weights from imagenet
        "include_top" implies if we need a classifer but not necessary since this is a segmentation task

    """
    encoder = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    # encoder = tf.keras.applications.ResNet50(weights='./resnet-weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_tensor=inputs)

    image_features = encoder.get_layer("conv4_block6_out").output

    """ Apply ASPP on the input features and get the output Tensor"""
    x_a = ASPP(image_features)
    """ The output tensor (feature map) from the ASPP encoder (see image "deeplabv3-architechture.png") should be upsampled 4 times """
    x_a = tf.keras.layers.UpSampling2D((4, 4 ), interpolation='bilinear')(x_a)
    
    ''' Now access the low-level features for the decoder part, which can be taken as an output of the encoder'''
    x_b = encoder.get_layer("conv2_block2_out").output
    ''' Apply the convolution layer  '''
    x_b = tf.keras.layers.Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)    

    ''' Combine the encoder and decoder to create the final 3 x 3 Conv'''
    x = tf.keras.layers.Concatenate()([x_a, x_b])
    x = SqueezeAndExcite(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = SqueezeAndExcite(x)
    
    ''' Now apply the upsampling by (4, 4) (see image "deeplabv3-architechture.png") '''
    x = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
    x = tf.keras.layers.Activation('sigmoid') (x)
    
    
    model = tf.keras.models.Model(inputs, x)

    return model


if __name__ == '__main__':
    model = deeplabv3_plus((512, 512, 3))
    # model.summary()