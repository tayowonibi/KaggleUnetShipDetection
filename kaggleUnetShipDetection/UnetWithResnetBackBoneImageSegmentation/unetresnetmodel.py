from keras import backend as K
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization, Activation, Conv2D, UpSampling2D, Conv2DTranspose,SpatialDropout2D
from keras import layers
from keras import Model


from keras.utils.data_utils import get_file

from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


import numpy as np

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMG_SCALING = (1, 1)



class UnetResnetModel:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, inputShape =(768,768,3), weightsPath=WEIGHTS_PATH, weightsPathNoTop=WEIGHTS_PATH_NO_TOP, isUnetWeightTrainable=False ):
        """  """
        
        self.resnet_weights_path_no_top =weightsPathNoTop
        self.resnet_weights_path =weightsPath
        self.unet_trainable = isUnetWeightTrainable
        self.unet_resnet_model =  self.get_unet_resnet(input_shape=inputShape )
    
    def conv_block_simple(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = BatchNormalization(name=prefix + "_bn")(conv)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv
    
    def conv_block_simple_no_bn(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv

    def ResNet50( self,include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        model = Model(img_input, x, name='resnet50')

        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    self.resnet_weights_path,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    self.resnet_weights_path_no_top,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path,by_name=True)
        return model


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'keras.., current block label, used for generating layer names
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

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'keras.., current block label, used for generating layer names
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

        x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    
    def get_unet_resnet(self, input_shape=None, input_tensor=None, unetWeightTrainable =None): 
        K.clear_session()
        
        if unetWeightTrainable is None:
            unetWeightTrainable = self.unet_trainable
        if input_tensor is None:
            resnet_base = self.ResNet50(input_shape=input_shape,input_tensor=None, include_top=False)
        else :
            resnet_base = self.ResNet50(input_shape=None, input_tensor =input_tensor, include_top=False)
        for l in resnet_base.layers:
            l.trainable = unetWeightTrainable
            if l.name.startswith("bn"):
                l.trainable=True
        conv1 = resnet_base.get_layer("activation_1").output
        conv2 = resnet_base.get_layer("activation_10").output
        conv3 = resnet_base.get_layer("activation_22").output
        conv4 = resnet_base.get_layer("activation_40").output
        conv5 = resnet_base.get_layer("activation_49").output

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.conv_block_simple(up7, 192, "conv7_1")
        conv7 = self.conv_block_simple(conv7, 192, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.conv_block_simple(conv9, 64, "conv9_2")

        up10 = UpSampling2D()(conv9)
        conv10 = self.conv_block_simple(up10, 32, "conv10_1")
        conv10 = self.conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.2)(conv10)
    
        x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
        #x = Conv2D(1, (1, 1), activation=None, name="prediction")(conv10)
        model = Model(resnet_base.input, x)

        return model
    
    def getModel (self):
        return self.unet_resnet_model
        

############################################################
#  Optimizers
############################################################

def dice_coef(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred, eps = 1e-6):
    return (K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred))) + eps)/(K.sum(y_true)+ (3*eps))


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)    



############################################################
#  Training Model
############################################################

def trainModel (unetModel, weight_path, train_data_generator, validation_data_generator , callback_list):

    
    loss_history1 = [unetModel.fit_generator(train_data_generator, 
                             steps_per_epoch=200, 
                             epochs=5, 
                             validation_data= validation_data_generator,
                             callbacks=callbacks_list,
                            validation_steps = 1,
                             workers=1 # the generator is not very thread safe
                                       )]

    for l in unetModel.layers:
        l.trainable = True
        
    #train all
    loss_history = [unetModel.fit_generator(train_data_generator, 
                             steps_per_epoch=2, 
                             epochs=2, 
                             validation_data= validation_data_generator,
                             callbacks=callbacks_list,
                            validation_steps = 2,
                             workers=1 
                                       )]
    return loss_history