import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D, Lambda, Input, BatchNormalization, GlobalAveragePooling2D
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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(num_classes,input_shape = (32, 32, 3)):
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = input_shape[0]  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 4
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    inputs = layers.Input(shape=input_shape)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    #flat = layers.Flatten()(features)
    logits = layers.Dense(num_classes, activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

class Network:
    def add_official_models(self):
        def input_for(image_width, image_height, channels):
            inputs = layers.Input(shape=(int(image_height), int(image_width), int(channels)))
            if int(channels) == 1:
                adjust_channels = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
            else:
                adjust_channels = inputs
            return inputs, adjust_channels

        def top_for(inputs, model, classes):
            # Freeze the pretrained weights
            model.trainable = True

            x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            x = layers.BatchNormalization()(x)

            top_dropout_rate = 0.2
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(len(classes), activation="softmax", name="pred")(x)

            return keras.Model(inputs, outputs)
        
        def convnextbase(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import ConvNeXtBase
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.convnext.preprocess_input(adjust_channels)
            model = ConvNeXtBase(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["convnextbase"] = convnextbase

        def inceptionv3(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import InceptionV3
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.inception_v3.preprocess_input(adjust_channels)
            model = InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["inceptionv3"] = inceptionv3

        def nasnetmobile(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import NASNetMobile
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.nasnet.preprocess_input(adjust_channels)
            
            model = NASNetMobile(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )

            return top_for(inputs, model, classes)
        
        self.versions["nasnetmobile"] = nasnetmobile
        
        def vgg16(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import VGG16
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.vgg16.preprocess_input(adjust_channels)
            model = VGG16(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["vgg16"] = vgg16

        def vgg19(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import VGG19
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.vgg19.preprocess_input(adjust_channels)
            model = VGG19(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["vgg19"] = vgg19

        def mobilenetv3small(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import MobileNetV3Small
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.mobilenet_v3.preprocess_input(adjust_channels)
            model = MobileNetV3Small(
                input_shape=None,
                alpha=1.0,
                minimalistic=True,
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                classes=len(classes),
                pooling=None,
                dropout_rate=0.2,
                classifier_activation="softmax",
                include_preprocessing=False
            )
            return top_for(inputs, model, classes)
        
        self.versions["mobilenetv3small"] = mobilenetv3small

        def mobilenetv3large(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import MobileNetV3Large
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.mobilenet_v3.preprocess_input(adjust_channels)
            model = MobileNetV3Large(
                input_shape=None,
                alpha=1.0,
                minimalistic=False,
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                classes=len(classes),
                pooling=None,
                dropout_rate=0.2,
                classifier_activation="softmax",
                include_preprocessing=False
            )
            return top_for(inputs, model, classes)
        
        self.versions["mobilenetv3large"] = mobilenetv3large

        def xception(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import Xception
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.xception.preprocess_input(adjust_channels)
            model = Xception(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["xception"] = xception

        def resnet50v2(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import ResNet50V2
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.resnet.preprocess_input(adjust_channels)
            model = ResNet50V2(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["resnet50v2"] = resnet50v2

        def efficientnetb0(model, image_width, image_height, classes, channels):
            from tensorflow.keras.applications import EfficientNetB0
            inputs, adjust_channels = input_for(image_width, image_height, channels)
            adjust_channels = keras.applications.efficientnet.preprocess_input(adjust_channels)
            model = EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_tensor=adjust_channels,
                input_shape=None,
                pooling=None,
                classes=len(classes),
                classifier_activation="softmax",
            )
            return top_for(inputs, model, classes)
        
        self.versions["efficientnetb0"] = efficientnetb0

        self.array_of_all_official_models = ["convnextbase", "vgg16", "vgg19", "mobilenetv3small", "mobilenetv3large", "xception", "resnet50v2", "efficientnetb0", "inceptionv3", "nasnetmobile"]

    def initialize_versions(self):
        self.versions = {}

        self.add_official_models()

        def v119(model, image_width, image_height, classes, channels):
            #model = create_vit_classifier(num_classes=len(classes), input_shape=(int(image_width),int(image_height),int(channels)))

            
            #model.add(Flatten())
            #model.add(Dense(len(classes), activation='softmax'))
            #return model
        
            model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),int(channels))))
            #model.add(Lambda(lambda x: tf.keras.applications.mobilenet.preprocess_input(tf.repeat(tf.cast(x, tf.float32), 3, axis=-1)),
            #                 input_shape=(int(image_width),int(image_height),int(channels))))
            #from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
            #base_model = MobileNetV3Small(weights='imagenet', input_shape=(int(image_width),int(image_height),3), include_top=False,
            ##                              include_preprocessing=False, classes=len(classes))

            # Freeze the pre-trained weights
            #for layer in base_model.layers:
            #    layer.trainable = False

            #model.add(base_model)

            model.add(Convolution2D(256, 5, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            model.add(Convolution2D(128, 5, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            '''model.add(Convolution2D(32, 5, strides=2, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            #model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))'''

            model.add(Convolution2D(32, 5, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))

            model.add(Convolution2D(128, 5, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(SpatialDropout2D(0.05))


            model.add(Dense(1024))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            #model.add(SpatialDropout2D(0.05))
            
            model.add(Flatten())
            model.add(Dense(len(classes), activation='softmax'))


            #from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
            #base_model = MobileNetV3Small(weights=None, input_shape=(int(image_width),int(image_height),int(channels)), include_top=True,
            #                              include_preprocessing=True, classes=len(classes))
            # Load pre-trained MobileNetV2, but without the final classification layers
            #base_model = MobileNetV2(weights='imagenet', include_top=False, 
            #                        input_shape=(int(image_width),int(image_height),3))

            # Freeze the pre-trained weights
            #for layer in base_model.layers:
            #    layer.trainable = False


            #i = tf.keras.layers.Input([int(image_width),int(image_height), int(3)], dtype = tf.uint8)
            #x = tf.cast(i, tf.float32)
            #x = tf.keras.applications.mobilenet.preprocess_input(x)
            # Add a normalization layer
            #model.add(Lambda(lambda x: tf.keras.applications.mobilenet.preprocess_input(tf.repeat(tf.cast(x, tf.float32), 3, axis=-1)),
            #                 input_shape=(int(image_width),int(image_height),int(channels))))


            #model.add(Lambda(lambda x: tf.repeat(x, 3, axis=-1), input_shape=(int(image_width),int(image_height),int(channels))))
            # model.add(x)
            # Add the base model
            #model.add(base_model)

            #model.add(GlobalAveragePooling2D())

            # Add a flattening layer
            #model.add(Flatten())

            # Add a dense layer with the number of classes as the number of nodes, with softmax activation for multiclass classification
            #model.add(Dense(len(classes), activation='softmax'))

            return model

        self.versions["v119"] = v119

        def v118_3(model, image_width, image_height, classes, channels):
            model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),int(channels))))

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

            return model

        self.versions["v118_3"] = v118_3


        def v110(model, image_width, image_height, classes, channels):
            model.add(Input(shape=(int(image_height),int(image_width),int(channels)), dtype=float))
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

        def v100(model, image_width, image_height, classes, channels):
            model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),int(channels))))
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

    def __init__(self, image_width, image_height, classes, learning_rate, version="current", channels=1):
        TRex.log("initializing network:"+str(image_width)+","+str(image_height)+" "+str(len(classes))+" classes with version "+ version)

        self.initialize_versions()
        if version == "current":
            version = "v119"

        model = Sequential()
        model = self.versions[version](model, image_width, image_height, classes, channels)

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