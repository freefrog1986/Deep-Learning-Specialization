import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
from scipy import misc
import keras.backend as K

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #print('Input image shape input_tensor:', input_tensor)
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    #print('Input image shape filter1:', x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    #print('Input image shape filter2:', x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #print('Input image shape filter3:', x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    #print('Input image shape last:', x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #print('Input image shape input_tensor:', input_tensor)
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    #print('Input image shape Conv2D-1:', x.shape)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    #print('Input image shape Conv2D-2:', x.shape)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #print('Input image shape Conv2D-3:', x.shape)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #print('Input image shape shortcut:', shortcut.shape)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    #print('Input image shape last:', x.shape)
    return x

def ResNet50():
    input_shape = (224,224,3)
    img_input = Input(shape=input_shape)
    bn_axis = 3

    print(img_input.shape)
    x = ZeroPadding2D((3, 3))(img_input)
    #print('Input image shape after zeropadding:', x.shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    #print('Input image shape after conv2d:', x.shape)

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    #print('Input image shape after bn:', x.shape)

    x = Activation('relu')(x)
    #print('Input image shape after activation:', x.shape)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    print('Input image shape after maxpooling2d:', x.shape)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    print('Input image shape after conv_block:', x.shape)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    print('Input image shape after identity_block:', x.shape)

    model = Model(img_input, x, name='resnet50')

    return model

model = ResNet50()

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#print('Input image shape:', x.shape)

preds = model.predict(x)

#print('Predicted:', decode_predictions(preds))