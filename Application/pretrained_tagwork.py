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
import TRex
import numpy as np

tagwork = None
model_path = None

class Tagwork:
    def __init__(self, width, height, model_path):
        self.width = width
        self.height = height
        self.counter = 0
        self.model_path = model_path
        #TRex.log("# image dimensions: "+str(self.width)+"x"+str(self.height))

    def load(self, path):
        from tensorflow import keras
        self.model = keras.models.load_model(path)

    def predict(self, images):
        assert self.model
        images = 255 - np.array(images, dtype=float)
        y = np.argmax(self.model.predict(images, verbose=0), axis=-1)
        #file = "/Users/tristan/Videos/locusts/samples/images_"+str(self.counter)+".npz"
        #print("saving to file", file);
        #np.savez(file, images=np.array(images), y=np.array(y));
        self.counter += 1
        return  y

def init():
    global width, height, tagwork, model_path
    #TRex.log("# initializing")
    tagwork = Tagwork(width, height, model_path)
    TRex.log("# loading network "+model_path)
    tagwork.load(model_path)
    
    TRex.log("# predicting with shape "+str(width)+"x"+str(height))
    images = np.zeros((100,height,width,1), dtype=float)
    TRex.log(str(images.shape))
    y = np.argmax(tagwork.model.predict(images))
    TRex.log(str(y))

def load():
    pass

def predict():
    global tagwork, tag_images, receive
    assert type(tagwork) != type(None)

    if len(tag_images) == 0:
        print("# empty images array")
    else:
        #print("# predicting ", len(tag_images))
        receive(tagwork.predict(tag_images).astype(np.int64))

        #del tag_images

