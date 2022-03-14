import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D, Lambda, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
#import TRex
import numpy as np

tagwork = None

class Tagwork:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.counter = 0
        #TRex.log("# image dimensions: "+str(self.width)+"x"+str(self.height))

    def load(self, path):
        from tensorflow import keras
        self.model = keras.models.load_model(path)

    def predict(self, images):
        assert self.model
        images = np.array(images, dtype=float)
        y = np.argmax(self.model.predict(images), axis=-1)
        file = "/Users/tristan/Videos/locusts/samples/images_"+str(self.counter)+".npz"
        print("saving to file", file);
        np.savez(file, images=np.array(images), y=np.array(y));
        self.counter += 1
        return  y

def init():
    global width, height, tagwork
    #TRex.log("# initializing")
    tagwork = Tagwork(width, height)
    #Trex.log("# loading network")
    tagwork.load("/Users/tristan/Videos/locusts/pretrained.h5")

def load():
    pass

def predict():
    global tagwork, tag_images, receive
    assert type(tagwork) != type(None)

    if len(tag_images) == 0:
        print("# empty images array")
    else:
        print("# predicting ", len(tag_images))
        receive(tagwork.predict(tag_images))

        #del tag_images

