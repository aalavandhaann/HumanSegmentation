# Tutorial source: [youtube link](https://www.youtube.com/watch?v=4LhUpCWBzT8)

## My questions

### I now understand a significant part of neural networks related to segmentation tasks

### However,

- [dataprocessing.py](./dataprocessing.py)
    - Vaidate my understanding
        - Data augmentation is a way to multiple the current inputs and also broaden the model to generalize well when training
        - Data augmentation is also useful to avoid overfitting or avoid bias?
- [model.py](./model.py)
    - See [image](./deeplabv3-architechture.png)
    - Well I now understand how they see the picture of a model (deeplab_v3 in our case) and implement the layers from the encoder and decoder parts
    - However, how did he know which **block** to get from the encoder?
        - For example, line 94 says 
            * ```image_features = encoder.get_layer("conv4_block6_out").output```
        - Another example, line 102 says 
            * ```x_b = encoder.get_layer("conv2_block2_out").output```
        - Should I read the paper to know these details?
        - Does it change between tensorflow and pytorch?
    - Do the box colors(blue, pink, brown, green) in the [picture of the model](./deeplabv3-architechture.png) mean anything?    
        * If you look at from ```lines 103 to 124``` the video tutor does a sequence of steps like
            * Conv2d
            * BatchNormalization
            * Activation
            * Is this something based on the color of the box? Are they some kind of legend keys in neural networks domain?
        * I ask this question because if you see the ```lines 61 to 68``` on image pooling (the last layer in the encoder part - see [image](./deeplabv3-architechture.png)) the applied steps are
            * AveragePooling2D
            * and then the usual steps applied to other tensors
                * Conv2d
                * BatchNormalization
                * Activation