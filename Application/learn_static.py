import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D, Lambda, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from visual_identification_network import Network

import numpy as np
import TRex

channels = 1

def reinitialize_network():
    global channels
    global model, image_width, image_height, classes, learning_rate, sess, network_version
    model = Network(image_width=image_width, image_height=image_height, classes=classes, learning_rate=learning_rate, version=network_version, channels=channels).model

class UserCancelException(Exception):
    """Raised when user clicks cancel"""
    pass
class UserSkipException(Exception):
    """Raised when user clicks cancel"""
    pass

class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, classes, X_test, Y_test, epochs, filename, prefix, output_path, compare_acc, estimate_uniqueness, settings):
        import numpy as np
        self.classes = classes
        self.model = model
        self.prefix = prefix
        self.output_path = output_path
        self.compare_acc = compare_acc
        self.estimate_uniqueness = estimate_uniqueness
        
        X = []
        Y = []
        for c in classes:
            mask = np.argmax(Y_test, axis=1) == c
            a = X_test[mask]
            X.append(a)
            Y.append(Y_test[mask])
        
            #TRex.log(len(a), c)
        
        self.X_test = X
        self.Y_test = Y
        self.epoch = 0
        self.batches = 0
        self.minimal_loss_allowed = 1
        self.uniquenesses = []
        self.worst_values = []
        self.mean_values = []
        self.better_values = []
        self.much_better_values = []
        self.per_class_accuracy = {}
        self.losses = []
        self.loss_diffs = []
        self.epochs = epochs
        self.best_result = {"weights":{}, "unique":-1}
        self.filename = filename
        self.settings = settings
        self.last_skip_step = np.inf
        
    def plot_comparison_raw(self, do_plot = True, length = -1):
        X_test = self.X_test
        Y_test = self.Y_test
        classes = self.classes
    
        matrix = np.zeros_like(np.ndarray(shape=(len(classes),len(classes))), dtype=float)
        result = np.zeros_like(np.ndarray(shape=(len(classes), 4), dtype=float))
        
        predictions = []

        for i, c, images, zeros in zip(np.arange(len(classes)), classes, X_test, Y_test):
            if len(images) == 0:
                continue

            Y = self.model.predict(tf.cast(images, dtype=np.float32), verbose=0)
            predictions.append(Y)
            
            distance = np.abs(Y - zeros).sum(axis=1)

            result[i, 0] = np.median(np.argmax(Y, axis=1))
            result[i, 1] = distance.std()
            result[i, 2] = np.median(Y.sum(axis=1)) #distance.mean()
            result[i, 3] = (np.argmax(Y, axis=1) == i).sum() / len(Y)
        return result, predictions
    
    def update_status(self, print_out = False, logs = {}):
        description = "epoch "+str(min(self.epoch+1, self.epochs))+"/"+str(self.epochs)
        if np.shape(self.X_test)[-1] > 0 and len(self.worst_values) > 0:
            description += " -- worst acc/class: {0:.3f}".format(self.worst_values[-1])

        if self.best_result["unique"] > -1:
            description = description + " best: "+str(float(int(self.best_result["unique"] * 10000)) / 100) + "%"
        if self.compare_acc > 0:
            description += " compare_acc: " + str(float(int(self.compare_acc * 10000)) / 100)+"%"
        if len(self.losses) >= 5:
            description += " loss_diff: " + str(float(int(abs(np.mean(self.loss_diffs[-5:])) / self.minimal_loss_allowed * 10000)) / 100)+"% of minimum"
        
        update_work_description(description)
        description = "[TRAIN] "+description

        if print_out:
            TRex.log(description+" "+str(logs))

    def evaluate(self, epoch, save = True, logs = {}):
        classes = self.classes
        model = self.model
        
        global update_work_percent, set_stop_reason, set_per_class_accuracy, set_uniqueness_history
        
        update_work_percent(min(epoch + 1, self.epochs) / self.epochs)
        
        self.epoch = min(epoch + 1, self.epochs)

        if np.shape(self.X_test)[-1] > 0:
            result, predictions = self.plot_comparison_raw(do_plot = False, length = -1)
            
            for i in range(0, len(result[:, 3])):
                if not i in self.per_class_accuracy:
                    self.per_class_accuracy[i] = []
                self.per_class_accuracy[i].append(result[:, 3][i])
            self.mean_values.append(np.mean(result[:, 3]))
            self.worst_values.append(np.min(result[:, 3]))

            set_per_class_accuracy(result[:, 3].astype(np.float))
            worst_acc_per_class = np.min(result[:, 3])
        else:
            result = None
            worst_acc_per_class = -1
        
        unique = self.estimate_uniqueness()

        if not save:
            return unique

        self.uniquenesses.append(unique)
        set_uniqueness_history(self.uniquenesses)

        if unique >= acceptable_uniqueness() and self.settings["accumulation_step"] >= -1:
            if self.settings["accumulation_step"] == -1:
                self.model.stop_training = True
                set_stop_reason("Uniqueness is sufficient ("+str(unique)+")")
            elif unique >= accepted_uniqueness():
                self.model.stop_training = True
                set_stop_reason("Uniqueness is sufficient ("+str(unique)+")")

        # check whether our worst value improved, but only use it if it wasnt the first epoch
        if unique > self.best_result["unique"] and (self.compare_acc <= 0 or unique >= self.compare_acc**2):
            self.best_result["unique"] = unique
            self.better_values.append((epoch, unique))
            TRex.log("\t(saving) new best-worst-value: "+str(unique))
            if unique >= self.compare_acc:
                self.much_better_values.append((epoch, unique))

            weights ={}
            for i, layer in zip(range(len(model.layers)), model.layers):
                h=layer.get_weights()
                #TRex.log(i, len(h), layer.get_config()["name"])
                if len(h) > 0:
                    weights[i] = h
            
            global output_path
            self.best_result["weights"] = weights
            try:
                TRex.log("\t(saving weights as '"+output_path+"_progress.npz')")
                np.savez(output_path+"_progress.npz", weights = np.array(weights, dtype="object"))
            except Exception as e:
                TRex.warn(str(e))
        else:
            TRex.log("\t(not saving) old best value is "+str(self.best_result["unique"])+" / "+str(unique))

        # not reaching a good enough result in N epochs
        if self.compare_acc > 0 and len(self.uniquenesses) >= 5 and self.best_result["unique"] < self.compare_acc**2:
            set_stop_reason("uniqueness stayed below "+str(int((self.compare_acc**2) * 1000) / 100.0)+"% in the first epochs")
            TRex.log("[STOP] best result is below "+str(self.compare_acc**2)+" even after "+str(epoch)+" epochs.")
            self.model.stop_training = True

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        if "val_acc" in logs:
            akey = "val_acc"
        else:
            akey = "acc"

        assert key in logs
        current_loss = logs[key]
        previous_loss = np.finfo(np.float).max
        
        if len(self.losses) > 0:
            l = np.nanmean(self.losses[-5:])
            if not np.isnan(l):
                previous_loss = l
        self.losses.append(current_loss)
        if len(self.losses) > 1:
            mu = np.mean(self.losses[-5:-1])
            self.loss_diffs.append((current_loss - mu))

        loss_diff = max(0.00000001, previous_loss - current_loss)
        self.minimal_loss_allowed = 0.05*10**(int(np.log10(current_loss))-1)
        change = np.diff(self.losses)

        if len(self.losses) > 1:
            TRex.log("\tminimal_loss_allowed: "+str(self.minimal_loss_allowed)+" -> mu: "+str(mu)+" current diffs: "+str(self.loss_diffs)+" average:"+str(np.mean(self.loss_diffs[-5:])))

        if not self.model.stop_training:
            #if self.compare_acc > -1 and "val_acc" in logs and len(self.worst_values) >= 5 and logs["val_acc"] < self.compare_acc**3:
            #    TRex.log("[STOP] val_acc is below "+str(self.compare_acc**3)+" even after "+str(len(self.worst_values))+" epochs.")
            #               self.model.stop_training = True

            # check for accuracy plateau
            long_time = int(max(5, self.epochs * 0.1))
            long_time = min(long_time, 10)
            TRex.log("-- worst_value "+str(self.worst_values[-2:])+" -- long time:"+str(long_time))
            if not self.model.stop_training and len(self.worst_values) >= 2 and self.settings["accumulation_step"] >= -1:
                acc = np.array(self.worst_values[-2:]) #logs[akey][-2:]
                if (acc > 0.97).all() or worst_acc_per_class >= 0.99:
                    TRex.log("[STOP] "+str(acc)+" in "+akey+" has been > 0.97 for consecutive epochs. terminating.")
                    set_stop_reason(akey+" good enough ("+str(acc)+")")
                    self.model.stop_training = True

            # check whether we are plateauing at a certain uniqueness level for a long time
            if not self.model.stop_training and len(self.uniquenesses) >= long_time and self.settings["accumulation_step"] > 0:
                acc = np.diff(self.uniquenesses[-long_time:]).mean() #logs[akey][-2:]
                TRex.log("Uniqueness plateau check:"+str(np.diff(self.uniquenesses[-long_time:]))+" -> "+str(acc))
                if acc <= 0.01:
                    set_stop_reason("uniqueness plateau")
                    TRex.log("[STOP] Uniqueness has been plateauing for several epochs. terminating. "+str(acc))
                    self.model.stop_training = True

            if not self.model.stop_training and len(self.losses) > 1:
                #if len(self.losses) >= 5 and np.abs(loss_diff) < minimal_loss_allowed:
                #                   TRex.log("[STOP] Loss is very small (epoch "+str(len(self.losses))+"). stopping. loss was "+str(current_loss)+" - "+str(previous_loss)+" = "+str(loss_diff))
                #                self.model.stop_training = True
    
                # check for loss plateau
                if not self.model.stop_training:
                    if len(self.losses) >= 5 and abs(np.mean(self.loss_diffs[-5:])) < self.minimal_loss_allowed:
                    #if len(self.losses) >= 5 and (np.array(self.loss_diffs[-2:]) < self.minimal_loss_allowed).all():
                        if self.settings["accumulation_step"] > 0 or (self.last_skip_step == self.settings["accumulation_step"] and len(self.uniquenesses) >= self.epochs * 0.5):
                            self.model.stop_training = True
                            set_stop_reason("small loss in consecutive epochs")
                            if self.settings["accumulation_step"] <= 0:
                                TRex.log("[STOP] Before second accumulation step, but #"+str(len(self.uniquenesses))+" epochs / "+str(self.epochs)+" is more than 50%.")
                            else:
                                TRex.log("[STOP] Loss is very small in consecutive epochs (epoch "+str(epoch)+"). stopping. loss was "+str(current_loss)+" vs. "+str(mu)+" "+str(self.loss_diffs[-2:]))
                        else:
                            TRex.log("(skipping small loss stopping criterion in first accumulation step)")
                            self.last_skip_step = self.settings["accumulation_step"]
                    elif len(change) >= 2:
                        TRex.log("\t"+str(current_loss)+" => "+str(current_loss - mu)+" ("+str(current_loss)+" / "+str(mu)+")")
                    count = 0
                    for i in range(-10,-1):
                        v = change[i:i+1]
                        if len(v) > 0 and v[0] >= 0:
                            count += 1
                        elif len(v) > 0:
                            count = 0
                    TRex.log("\tcounted "+str(count)+" increases in loss in consecutive epochs - "+str(change))
                    if count >= 4 and (self.settings["accumulation_step"] != 0 or (self.settings["accumulation_step"] == 0 and len(self.losses) >= 30)):
                        # we seem to have started overfitting
                        set_stop_reason("overfitting")
                        TRex.log("[STOP] overfitting. stopping with loss diffs: "+str(change))
                        self.model.stop_training = True

        self.update_status(True, logs=logs)
        self.batches = 0
        return unique
    
    def on_epoch_end(self, epoch, logs={}):
        worst_value = self.evaluate(epoch, True, logs)

        global get_abort_training, get_skip_step
        if get_abort_training():
            global UserCancelException
            self.model.stop_training = True
            #TRex.log("aborting because we have been asked to by main")
            raise UserCancelException()
        if get_skip_step():
            global UserSkipException
            self.model.stop_training = True
            raise UserSkipException()

    def on_batch_end(self, batch, logs={}):
        global get_abort_training, get_skip_step
        if get_abort_training():
            global UserCancelException
            self.model.stop_training = True
            raise UserCancelException()
        if get_skip_step():
            global UserSkipException
            self.model.stop_training = True
            raise UserSkipException()

        self.batches += 1
        if batch % 50 == 0:
            self.update_status(logs=logs)
            global update_work_percent
        epoch = self.epoch
        epoch += self.batches / self.settings["per_epoch"]
        update_work_percent((epoch) / self.epochs)
        #logs = logs or {}
        #batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        #num_steps = logs.get('num_steps', 1)
        #if self.use_steps:
        #    self.seen += num_steps
        #else:
        #    self.seen += batch_size * num_steps

        #self.display_step += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        #if self.verbose and self.seen < self.target and self.display_step % #self.display_per_batches == 0:
        #    self.progbar.update(self.seen, self.log_values)

def r3g3b2_to_vec(r3g3b2_array, channels=3, alpha=255):
    if channels == 3:
        return np.stack([
            ((r3g3b2_array >> 6) & 3) * 64,
            ((r3g3b2_array >> 3) & 7) * 32,
            (r3g3b2_array & 7) * 32
        ], axis=-1).astype(np.uint8)
    elif channels == 4:
        return np.stack([
            ((r3g3b2_array >> 6) & 3) * 64,
            ((r3g3b2_array >> 3) & 7) * 32,
            (r3g3b2_array & 7) * 32,
            np.full(r3g3b2_array.shape, alpha)
        ], axis=-1).astype(np.uint8)
    else:
        raise ValueError("Unsupported number of channels: %d" % channels)

def load_weights():
    global output_path, model
    try:
        with np.load(output_path+'.npz', allow_pickle=True) as npz:
           m = npz['weights'].item()
           for i, layer in zip(range(len(model.layers)), model.layers):
               if i in m:
                   layer.set_weights(m[i])
    except Exception as e:
        TRex.warn(str(e))
        raise Exception("Failed to load weights. This is likely because either the individual_image_size is different, or because existing weights might have been generated by a different version of TRex (see visual_identification_version).")

def predict():
    global receive, images, model, channels
    
    if channels == 3:
        train_X = r3g3b2_to_vec(np.array(images, copy=False)[..., 0])
    else:
        train_X = np.array(images, copy=False)
    if len(train_X.shape) != 4:
        print("error with the shape ", train_X.shape, " < len 4")
    elif train_X.shape[1] != model.input.shape[1] or train_X.shape[2] != model.input.shape[2]:
        raise Exception("Wrong image dimensions for model ("+str(train_X.shape[1:3])+" vs. "+str(model.input.shape[1:3])+").")
        
    indexes = np.array(np.arange(len(train_X)), dtype=np.float32)
    output = model.predict(tf.cast(train_X, dtype=np.float32), verbose=0)
    
    receive(output, indexes)

    del output
    del train_X
    del indexes
    del images

def start_learning():
    global channels
    global best_accuracy_worst_class, max_epochs, image_width, image_height, update_work_percent
    global output_path, classes, learning_rate, accumulation_step, global_segment, verbosity
    global batch_size, X_val, Y_val, X, Y, run_training, save_weights_after, do_save_training_images, min_iterations

    #batch_size = int(max(batch_size * 2, 64))
    #if batch_size >= 128:
    #    batch_size = 200
    #else:
    batch_size = int(max(batch_size, 64))
    epochs = max_epochs
    #batch_size = 32
    move_range = min(0.05, 2 / min(image_width, image_height))
    TRex.log("setting move_range as "+str(move_range))

    settings = {
        "epochs": max_epochs,
        "batch_size": batch_size,
        "move_range": move_range,
        "output_path": output_path,
        "image_width": image_width,
        "image_height": image_height,
        "classes": np.array(classes,dtype=int),
        "learning_rate": learning_rate,
        "accumulation_step": accumulation_step,
        "global_segment": np.array(global_segment, dtype=int),
        "per_epoch" : -1,
        "min_iterations": min_iterations,
        "verbosity": verbosity
        #"min_acceptable_value": 0.98
    }

    if len(X_val) == 0 or channels != 3:
        X_test = np.array(X_val, copy=False)
    else:
        X_test = r3g3b2_to_vec(np.array(X_val, copy=False)[..., 0]) #True, dtype = float) / 255.0
    Y_test = np.array(Y_val, dtype=float)
    original_classes_test = np.copy(Y_test)
    Y_test = to_categorical(Y_test, len(classes))

    if len(X) == 0 or channels != 3:
        X_train = np.array(X, copy=False)
    else:
        X_train = r3g3b2_to_vec(np.array(X, copy=False)[..., 0])
    Y_train = np.array(Y, dtype=float)

    original_classes = np.copy(Y_train)
    #test_original_classes = np.copy(test_Y)
    Y_train = to_categorical(Y_train, len(classes))
    #Y_test = np_utils.to_categorical(test_Y, len(classes))

    #X_train = tf.constant(X_train, dtype=float)
    #Y_train = tf.constant(Y_train, dtype=float)

    #X_test = tf.constant(X_test, dtype=float)
    #Y_test = tf.constant(Y_test, dtype=float)

    print("Python received "+str(X_train.shape)+" training images ("+str(Y_train.shape)+") and  "
          +str(X_test.shape)+" validation images ("+str(Y_test.shape)+")")

    mi = 0
    for i, c in zip(np.arange(len(classes)), classes):
        mi = max(mi, len(Y_train[np.argmax(Y_train, axis=1) == c]))

    per_epoch = max(settings["min_iterations"], int(len(X_train) // batch_size))# * 2.0) # i am using augmentation
    per_epoch = int((per_epoch // batch_size) * batch_size)
    settings["per_epoch"] = per_epoch
    TRex.log(str(settings))

    per_class = {}
    m = 0
    cvalues = []
    for i, c in zip(np.arange(len(classes)), classes):
        per_class[i] = len(Y_train[np.argmax(Y_train, axis=1) == c])
        cvalues.append(per_class[i])
        if per_class[i] > m:
            m = per_class[i]

    TRex.log("# [init] samples per class "+str(per_class))

    def preprocess(images):
        dtype = images.dtype
        assert images.max() <= 255 and images.max() > 1
        #TRex.log(images.shape, images.dtype)
        
        alpha = 1.0
        if len(images.shape) == 3:
            alpha = np.random.uniform(0.85, 1.15)
        else:
            alpha = []
            for i in range(len(images)):
                alpha.append(np.random.uniform(0.85, 1.15))
            alpha = np.array(alpha)
        
        images = images.astype(np.float) * alpha
        
        return np.clip(images, 0, 255).astype(dtype)

    datagen = ImageDataGenerator(#rescale = 1.0/255.0,
                                 rotation_range = 180,
                                 #zoom_range = 0.1,
                                 #horizontal_flip = True,
                                 #vertical_flip = True,
                                 #brightness_range=(0.5,1.5),
                                 width_shift_range=move_range,
                                 height_shift_range=move_range,
                                 cval = 0,
                                 fill_mode = "constant",
                                 #preprocessing_function=preprocess,
                                 #use_multiprocessing=True
                                 )

    cvalues = np.array(cvalues)

    if len(X_train) > 0 and (channels == 3 or channels == 1):
        # Initialize the batch counter and the numpy array
        grid_shape = (2, 10)
        image_shape = X_train.shape[1:3]  # shape of a single image
        print(image_shape)
        grid = np.zeros((grid_shape[0]*image_shape[0], grid_shape[1]*image_shape[1], channels))

        _input = X_train
        for y in range(grid_shape[0]):
            for x in range(grid_shape[1]):
                # Generate augmented images
                for batch in datagen.flow(_input, batch_size=1):
                    # Add to grid
                    print(_input.shape, batch[0].shape, grid_shape, image_shape)
                    pos = (x * image_shape[1], y * image_shape[0])
                    dim = ((x+1) * image_shape[1], (y+1) * image_shape[0])
                    grid[pos[1]:dim[1], pos[0]:dim[0]] = batch[0, ...].astype(np.uint8)

                    #grid[(i//grid_shape[0]*image_shape[0]):((i//grid_shape[0] + 1)*image_shape[0]), 
                    #    (i%grid_shape[1]*image_shape[1]):((i%grid_shape[1] + 1)*image_shape[1])] = 255 - batch[0].astype(np.uint8)
                    break  # We only want one augmentation per image, so break the loop
        
        print(grid.shape, grid.dtype)
        import cv2 as cv
        if channels == 1:
            grid = cv.cvtColor(grid.astype(np.uint8), cv.COLOR_GRAY2RGBA)
        else:
            grid = cv.cvtColor(grid.astype(np.uint8), cv.COLOR_RGB2RGBA)
        TRex.imshow("grid", grid.astype(np.uint8))

    TRex.log("# [init] weights per class "+str(per_class))
    TRex.log("# [training] data shapes: train "+str(X_train.shape)+" "+str(Y_train.shape)+" test "+str(Y_test.shape)+" "+str(classes))
    TRex.log("# [values] X:"+str(tf.reduce_max(X_train).numpy())+" - "+str(tf.reduce_min(X_train).numpy()))
    TRex.log("# [values] Y:"+str(tf.reduce_max(Y_train).numpy())+" - "+str(tf.reduce_min(Y_train).numpy()))
    if do_save_training_images():
        try:
            np.savez(output_path+"_training_images.npz", X_train=X_train, Y_train=Y_train, classes=classes)
        except:
            pass

    abort_with_error = False

    try:
        if run_training:
            update_work_percent(0.0)

            callback = ValidationCallback(model, classes, X_test, Y_test, epochs, filename, output_prefix+"_"+str(accumulation_step), output_path, best_accuracy_worst_class, estimate_uniqueness, settings)
            
            validation_data = None
            #validation_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_test, float), Y_test))#.batch(batch_size)
            if len(X_test) == 0:
                validation_data = None
            else:
                validation_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_test, float), Y_test)).batch(batch_size)

            dataset = tf.data.Dataset.from_generator(lambda: datagen.flow(tf.cast(X_train, float), tf.cast(Y_train, float), batch_size=batch_size), 
                output_types=(tf.float32, tf.float32),
                output_shapes =(tf.TensorShape([None, int(settings["image_height"]), int(settings["image_width"]), int(channels)]), tf.TensorShape([None, int(len(classes))]))
            ).repeat().shuffle(batch_size * 2, reshuffle_each_iteration=True)#.batch(batch_size)
            
            #dataset = datagen.flow(tf.cast(X_train, float), Y_train, batch_size=batch_size)
            TRex.log("tf.data.Dataset: "+str(dataset))
            TRex.log("tf.data.Dataset (validation): "+str(validation_data))
            history = model.fit(dataset,
                                  validation_data=validation_data,
                                  steps_per_epoch=per_epoch, 
                                  epochs=max_epochs,
                                  callbacks=[callback],
                                  verbose=verbosity)
            
            '''validation_data = (X_test, Y_test)
            if len(X_test) == 0:
                validation_data = None

            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                          validation_data=validation_data,
                                          steps_per_epoch=per_epoch, epochs=epochs,
                                          callbacks = [ callback ],
                                          #class_weight = per_class
                                          )'''

            #model_json = model.to_json()
            #with open(output_path+".json", "w") as f:
            #    f.write(model_json)
            
            if callback.best_result["unique"] != -1:
                weights = callback.best_result["weights"]
                history_path = output_path+"_"+str(accumulation_step)+"_history.npz"
                TRex.log("saving histories at '"+history_path+"'")
                
                np.savez(history_path, history=np.array(history.history, dtype="object"), uniquenesses=callback.uniquenesses, better_values=callback.better_values, much_better_values=callback.much_better_values, worst_values=callback.worst_values, mean_values=callback.mean_values, settings=np.array(settings, dtype="object"), per_class_accuracy=np.array(callback.per_class_accuracy),
                    samples_per_class=np.array(per_class, dtype="object"))

                if save_weights_after:
                    np.savez(output_path+".npz", weights = np.array(weights, dtype="object"))
                    TRex.log("saved model to "+output_path+" with accuracy of "+str(callback.best_result["unique"]))
            
                try:
                    for i, layer in zip(range(len(model.layers)), model.layers):
                        if i in weights:
                            layer.set_weights(weights[i])

                    if len(X_train) > 0:
                        callback.evaluate(epochs, False)
                        best_accuracy_worst_class = callback.best_result["unique"]
                except:
                    TRex.warn("loading weights failed")
            elif settings["accumulation_step"] == -2:
                TRex.warn("could not improve upon previous steps.")
            else:
                TRex.warn("Could not improve upon previous steps.")
                abort_with_error = True

        else:
            # just run the test
            update_work_percent(1.0)
            
            callback = ValidationCallback(model, classes, X_test, Y_test, epochs, filename, output_prefix, output_path, best_accuracy_worst_class, estimate_uniqueness, settings)
            if len(X_train) > 0:
                callback.evaluate(0, False)

    except UserCancelException as e:
        #TRex.warn("exception during training: "+str(e))
        abort_with_error = True
        TRex.log("Ending training because user cancelled the action.")
    except UserSkipException as e:
        abort_with_error = False
        TRex.log("Ending training because user skipped the action.")

    del X_train
    del Y_train
    del X_test
    del Y_test
    del X
    del Y
    del callback
    del X_val
    del Y_val

    if "validation_data" in locals():
        del validation_data
    if "datagen" in locals():
        del datagen
    if "weights" in locals():
        del weights

    import gc
    gc.collect()

    if abort_with_error:
        raise Exception("aborting with error")
