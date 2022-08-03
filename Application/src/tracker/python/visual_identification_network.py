import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D, Lambda, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import TRex

# from https://github.com/umbertogriffo/focal-loss-keras
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

class Network:

    def initialize_versions(self):
        self.versions = {}

        def v118_3(model, image_width, image_height, classes):
            model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),1)))

            model.add(Convolution2D(16, 5))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            model.add(Convolution2D(64, 5))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            model.add(Convolution2D(100, 5))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            model.add(Dense(100))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.05))
            
            model.add(Flatten())
            model.add(Dense(len(classes), activation='softmax'))

        self.versions["v118_3"] = v118_3


        def v110(model, image_width, image_height, classes):
            model.add(Input(shape=(int(image_height),int(image_width),1), dtype=float))
            model.add(Lambda(lambda x: (x / 127.5 - 1.0)))
            
            model.add(Convolution2D(16, kernel_size=(5,5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.25))
            
            model.add(Convolution2D(64, kernel_size=(5,5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.25))
            #model.add(Dropout(0.5))

            model.add(Convolution2D(100, kernel_size=(5,5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.25))
            #model.add(Dropout(0.5))
            
            model.add(Dense(100))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.25))
            
            model.add(Flatten())
            model.add(Dense(len(classes), activation='softmax'))

            return model

        self.versions["v110"] = v110

        def v100(model, image_width, image_height, classes):
            model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),1)))
            model.add(Convolution2D(16, 5, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.25))

            model.add(Convolution2D(64, 5, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.25))

            model.add(Convolution2D(100, 5, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.25))

            model.add(Flatten())
            model.add(Dense(100, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(classes), activation='softmax'))

            return model

        self.versions["v100"] = v100

    def __init__(self, image_width, image_height, classes, learning_rate, version="current"):
        TRex.log("initializing network:"+str(image_width)+","+str(image_height)+" "+str(len(classes))+" classes with version "+ version)

        self.initialize_versions()
        if version == "current":
            version = "v118_3"

        model = Sequential()
        self.versions[version](model, image_width, image_height, classes)

        import platform
        import importlib
        found = True
        try:
            importlib.import_module('tensorflow')
            import tensorflow
        except ImportError:
            found = False

        if found:
            model.compile(loss= #'categorical_crossentropy',
                #SigmoidFocalCrossEntropy(),
                categorical_focal_loss(gamma=2., alpha=.25),
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])


        model.summary(print_fn=TRex.log)
        self.model = model