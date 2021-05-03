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
    def __init__(self, width, height, categories, output_file):
        self.categories_map = eval(categories)
        self.categories = [c for c in self.categories_map]

        self.width = width
        self.height = height

        self.update_required = False
        self.last_size = 0
        self.output_file = output_file

        self.samples = []
        self.labels = []
        self.validation_indexes = np.array([], dtype=int)

        TRex.log("# image dimensions: "+str(self.width)+"x"+str(self.height))
        TRex.log("# initializing categories "+str(categories))

        self.reload_model()

    def reload_model(self):
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

    def send_samples(self):
        global recv_samples

        TRex.log("# sending "+str(len(self.samples))+" samples")
        recv_samples(np.array(self.samples).astype(np.uint8).flatten(), self.labels)

    def add_images(self, images, labels):
        # length before adding images
        prev_L = len(self.labels)
        TRex.log("# previously had "+str(len(self.samples))+" images")

        for image in images:
            self.samples.append(image)

        for l in labels:
            self.labels.append(str(l));

        self.updated_data(prev_L, labels)

    def updated_data(self, prev_L, labels):
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
            self.validation_indexes = np.concatenate((self.validation_indexes, next_indexes[:missing]), axis=0).astype(int)
            TRex.log("# now have "+str(len(self.validation_indexes))+" validation samples and "+str(len(self.samples) - len(self.validation_indexes))+" training samples")

        TRex.log("# labels dist: "+str(per_class))
        if len(np.unique(self.labels)) == len(self.categories):
            if len(self.samples) - self.last_size >= 500:
                self.update_required = True
                TRex.log("# scheduling update. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))
                self.last_size = len(self.samples)
            else:
                TRex.log("# no update required. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))

    def load(self):
        self.reload_model()

        #try:
        with np.load(self.output_file, allow_pickle=True) as npz:
            shape = npz["x"].shape
            TRex.log("# loading model with data of shape "+str(shape)+" and current shape "+str(self.height)+","+str(self.width))
            assert shape[1] == self.height and shape[2] == self.width

            categories_map = npz["categories_map"].item()
            TRex.log("# categories_map:"+str(categories_map))
            categories = [c for c in categories_map]

            if categories != self.categories:
                TRex.log("# categories are different: "+str(categories)+" != "+str(self.categories)+". replacing current samples.")

                self.categories = categories
                self.categories_map = categories_map
                self.samples = []
                self.validation_indexes = np.array([], dtype=int)
                self.labels = []


            m = npz['weights'].item()
            for i, layer in zip(range(len(self.model.layers)), self.model.layers):
                if i in m:
                    layer.set_weights(m[i])

            validation_indexes = npz["validation_indexes"].astype(int)
            TRex.log("# loading indexes: "+str(validation_indexes))
            TRex.log("# adding data: "+str(npz["x"].shape))

            # add current offset to validation_indexes
            validation_indexes += len(self.samples)
            TRex.log("# with offset: "+str(validation_indexes))
            self.validation_indexes = np.concatenate((self.validation_indexes, validation_indexes), axis=0)
            
            # add data
            prev_L = len(self.labels)
            TRex.log("# unique new labels: "+str(np.unique(npz["y"])))

            for image in npz["x"]:
                self.samples.append(image)
            for y in npz["y"]:
                self.labels.append(str(y))

            self.updated_data(prev_L, [])

        if len(self.samples) > 0:
            X = np.array(self.samples)

            Y = np.zeros(len(self.labels), dtype=int)
            L = self.categories
            for i in range(len(L)):
                Y[np.array(self.labels) == L[i]] = self.categories_map[L[i]]
            Y = to_categorical(Y, len(L))

            X_test = X[self.validation_indexes]
            Y_test = Y[self.validation_indexes]
            self.model.evaluate(X_test, Y_test, batch_size=64)
            self.update_best_accuracy(X_test, Y_test)
        else:
            TRex.log("# no data available for evaluation")

        #except Exception as e:
        #    TRex.warn("loading weights failed: "+str(e))

    def perform_training(self):
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

        try:

            weights = {}
            for i, layer in zip(range(len(self.model.layers)), self.model.layers):
                h = layer.get_weights()
                #TRex.log(i, len(h), layer.get_config()["name"])
                if len(h) > 0:
                    weights[i] = h

            np.savez(self.output_file, weights = np.array(weights, dtype="object"), x=self.samples, y=self.labels, validation_indexes=self.validation_indexes, categories_map=np.array(self.categories_map, dtype='object'))
            TRex.log("# UPDATE: saved samples and weights.")

        except Exception as e:
            TRex.log("Saving weights and samples failed: "+str(e))

        self.update_best_accuracy(X_test, Y_test)

    def update_best_accuracy(self, X_test, Y_test):
        global set_best_accuracy
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
    global categorize, categories, width, height, output_file

    if type(categorize) == type(None):
        categorize = Categorize(width, height, categories, output_file)

    TRex.log("# initialized.")

def load():
    global categorize
    assert type(categorize) != type(None)

    if type(categorize) == type(None):
        start()
    else:
        TRex.log("# model already exists. reloading model")
        categorize.load()

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

def send_samples():
    global categorize
    categorize.send_samples()

def predict():
    global categorize, images, receive
    assert type(categorize) != type(None)

    receive(categorize.predict(images))

def clear_images():
    global categorize
    assert type(categorize) != type(None)

    TRex.log("# clearing images")
    categorize = Categorize(categorize.width, categorize.height, str(categorize.categories))

