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

categorize = None

class Categorize:
    def __init__(self, width, height, categories):
        self.categories_map = eval(categories)
        self.categories = [c for c in self.categories_map]

        self.width = width
        self.height = height

        self.update_required = False
        self.last_size = 0

        self.samples = []
        self.labels = []
        self.validation_indexes = np.array([], dtype=int)

        TRex.log("# image dimensions: "+str(self.width)+"x"+str(self.height))
        TRex.log("# initializing categories "+str(categories))

        self.model = Sequential()

        self.model.add(Input(shape=(int(self.height),int(self.width),1), dtype=float))
        self.model.add(Lambda(lambda x: (x / 127.5 - 1.0)))
        
        self.model.add(Convolution2D(16, kernel_size=(5,5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(SpatialDropout2D(0.25))
        
        self.model.add(Convolution2D(64, kernel_size=(5,5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(SpatialDropout2D(0.25))

        self.model.add(Convolution2D(100, kernel_size=(5,5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(SpatialDropout2D(0.25))
        
        self.model.add(Dense(100))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(SpatialDropout2D(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(len(self.categories), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr=0.001),
            metrics=['accuracy'])

        self.model.summary(print_fn=TRex.log)

    def add_images(self, images, labels):
        # length before adding images
        prev_L = len(self.labels)

        for image in images:
            self.samples.append(image)

        for l in labels:
            self.labels.append(str(l));

        TRex.log("# samples are "+str(np.shape(self.samples))+" labels:"+str(np.shape(self.labels)))
        TRex.log("# "+str(np.unique(self.labels)))

        per_class = {}
        for label in self.labels:
            per_class[label] = 0

        for i in range(len(self.labels)):
            per_class[self.labels[i]] += 1

        missing = int(0.33 * len(self.samples) - len(self.validation_indexes))
        if missing > 0:
            TRex.log("# missing "+str(missing)+" validation samples, adding...")
            next_indexes = np.arange(prev_L, len(labels) + prev_L, dtype=int)
            np.random.shuffle(next_indexes)
            self.validation_indexes = np.concatenate((self.validation_indexes, next_indexes[:missing]), axis=0, dtype=int)
            TRex.log("# now have "+str(len(self.validation_indexes))+" validation samples and "+str(len(self.samples) - len(self.validation_indexes))+" training samples")

        TRex.log("# labels dist: "+str(per_class))
        if len(np.unique(self.labels)) == len(self.categories):
            if len(self.samples) - self.last_size >= 500:
                self.update_required = True
                TRex.log("# scheduling update. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))
                self.last_size = len(self.samples)
            else:
                TRex.log("# no update required. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))

    def perform_training(self):
        global set_best_accuracy
        import numpy as np

        TRex.log("# performing training...")
        batch_size = 32

        Y = np.zeros(len(self.labels), dtype=int)
        L = self.categories

        for i in range(len(L)):
            Y[np.array(self.labels) == L[i]] = self.categories_map[L[i]]

        Y = to_categorical(Y, len(L))
        TRex.log("Y:"+str(Y))

        X = np.array(self.samples)

        training_indexes = np.arange(len(X), dtype=int)
        TRex.log("training:"+str(type(training_indexes))+" val:"+str(type(self.validation_indexes)))
        training_indexes = np.delete(training_indexes, self.validation_indexes)

        train_len = int(len(X) * 0.77)
        X_train = X[training_indexes]
        Y_train = Y[training_indexes]

        X_test = X[self.validation_indexes]
        Y_test = Y[self.validation_indexes]

        training_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_train, float), Y_train)).batch(batch_size)
        validation_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_test, float), Y_test)).batch(batch_size)

        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        self.model.fit(training_data, validation_data=validation_data, epochs=10, verbose=2, callbacks=[early_stopping_monitor])
        self.model.save('model')

        y_test = np.argmax(Y_test, axis=1) # Convert one-hot to index
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        report = classification_report(y_test, y_pred, output_dict=True)
        for key in report:
            TRex.log("report: "+str(key)+" "+str(report[key]))
        TRex.log(str(report))
        set_best_accuracy(float(report["accuracy"]))

    def update(self):
        if self.update_required:
            self.update_required = False
            TRex.log("# UPDATE: saving samples...")
            np.savez("training_data.npz", x=self.samples, y=self.labels)
            self.perform_training()

    def predict(self, images):
        assert self.model

        TRex.log(str(np.shape(images)))
        images = np.array(images, dtype=float)
        TRex.log(str(np.shape(images)))

        y = np.argmax(self.model.predict(images), axis=-1)
        TRex.log("Y:"+str(y))
        return  y


def start():
    global categorize, categories, width, height

    if type(categorize) == type(None):
        categorize = Categorize(width, height, categories)

    TRex.log("# initialized.")

def add_images():
    global categorize, additional, additional_labels
    assert type(categorize) != type(None)

    TRex.log("# adding "+str(len(additional))+" images")
    categorize.add_images(additional, additional_labels)

def post_queue():
    # called whenever the add_images things are over
    global categorize
    assert type(categorize) != type(None)

    categorize.update()

def predict():
    global categorize, images, receive
    assert type(categorize) != type(None)

    receive(categorize.predict(images))


