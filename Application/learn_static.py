#try:
#    from jupyterthemes import jtplot
#    jtplot.style()
#except:
#    pass

import matplotlib
matplotlib.use("Agg")

from keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D,MaxPooling2D
from keras.models import Sequential
import keras
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import TRex
import utils
import matplotlib.pyplot as plt

import shutil

from keras.layers import Dense, Dropout, Activation, Cropping2D, Flatten, Convolution1D, Convolution2D, MaxPooling1D,MaxPooling2D, Lambda
from keras.layers import SpatialDropout2D

#### NETWORK LAYOUT

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
        from keras import backend as K
        import tensorflow as tf

        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        from keras import backend as K
        import tensorflow as tf

        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

def reinitialize_network():
    global model, image_width, image_height, classes, learning_rate

    model = Sequential()
    TRex.log("initializing network:"+str(image_width)+","+str(image_height)+" "+str(len(classes))+" classes")

    model.add(Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(int(image_width),int(image_height),1)))
    model.add(Convolution2D(16, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.25))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.25))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(100, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.25))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    #### / NETWORK LAYOUT
    import platform
    import importlib
    found = True
    try:
        importlib.import_module('tensorflow')
        import tensorflow
    except ImportError:
        found = False

    if found:
        model.compile(loss=#'categorical_crossentropy',
            categorical_focal_loss(gamma=2., alpha=.25),
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            metrics=['accuracy'])
    model.summary(print_fn=TRex.log)


def plot_medians(title, train_X, train_Y, classes):
    lengths = {}
    f, axes = utils.axes_for_array(len(classes))
    for ax, c in zip(axes, classes):
        index = train_Y == c
        ax.set_title(str(c)+": "+str(len(train_X[index])))
        lengths[c] = len(train_X[index])
        
        subdata = np.copy(train_X[index])
        
        if len(subdata) == 0:
            #TRex.log("skipping class "+str(c)+" because its empty")
            continue
        
        img = np.max(subdata.astype(float), axis=0)[:,:,0]
        img = np.stack((img, )*3, axis=2).astype(np.uint8)
        ax.imshow(img)
        
        del subdata
        
    TRex.log("#samples per class: "+str(lengths))
    TRex.imshow(title, utils.figure_as_image())
    #utils.show_figure(title, output_path="")

class UserCancelException(Exception):
    """Raised when user clicks cancel"""
    pass

class ValidationCallback(keras.callbacks.Callback):
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

            Y = self.model.predict(images)
            predictions.append(Y)
            
            distance = np.abs(Y - zeros).sum(axis=1)

            if do_plot:
                f, [ax, bx] = plt.subplots(1, 2, figsize=(15, 5))
                ax.scatter(np.arange(len(distance)), distance, label="d", s = 1, c = np.ones(len(distance)), alpha = 0.5)
                pts = np.argmax(Y, axis=1)
                color = Y.sum(axis=1)

                for j in range(len(color)):
                    color[j] = color[j] * len(Y[j][Y[j] > 0.01])# / len(obj.classes)

                pts = np.concatenate((pts, [0, ]), axis=0)
                color = np.concatenate((color, [10, ]), axis=0)

                im = bx.scatter(np.arange(len(pts)), pts, label="Y", s = 1, c = color, alpha = 0.5)
                bx.set_yticks(np.arange(len(obj.classes_subset)))
                bx.set_ylim(-1, len(obj.classes_subset))
                #TRex.log(color.max(), color.min())
                plt.colorbar(im)
                plt.suptitle(c)
                ax.legend()
                bx.legend()
                plt.show()
                plt.close(f)

            result[i, 0] = np.median(np.argmax(Y, axis=1))
            result[i, 1] = distance.std()
            result[i, 2] = np.median(Y.sum(axis=1)) #distance.mean()
            result[i, 3] = (np.argmax(Y, axis=1) == i).sum() / len(Y)
        return result, predictions
        
    def plot_confusion_matrix(self, cm, classes,
                                      normalize,
                                      title,
                                      cmap, ax = None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        do_show = False
    
        if normalize:
            cm = cm.astype('float') / (1+cm.sum(axis=1)[:, np.newaxis])

        if ax == None:
            f, ax = plt.subplots(1,1)
            do_show = True
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax = ax)
        cbar_ticks = np.linspace(0., 1., num=10, endpoint=True)
        cbar.set_ticks(cbar_ticks)
        
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([str(c) for c in classes])

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
    
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        if do_show:
            utils.show_figure("confusion matrix", output_path="")
        #plt.tight_layout()
        
    def plot_confusion(self, X_test, Y_test, predictions = None,  title = "", ax = None, cmap = None):
        classes = self.classes
        #TRex.log("shapes:", X_test.shape, Y_test.shape)
        
        y = None
        if type(predictions) == type(None):
            y = self.model.predict(X_test)
        else:
            y = np.concatenate(predictions, axis=0)
            assert len(y) == len(X_test)
        
        result = np.zeros_like(np.ndarray(shape=(len(classes),len(classes))), dtype=float)

        wrong = 0
        correct = 0
        
        for i in range(len(X_test)):
            a = np.argmax(y[i,0:len(classes)])
            b = np.argmax(Y_test[i,0:len(classes)])

            if a == b:
                correct = correct+1
            else:
                wrong = wrong + 1

            result[a, b] = result[a, b] + 1
            #result_x.append(a)
            #result_y.append(b)

        title = title + " (accuracy "+str(1-wrong/float(correct+wrong))+")"
        self.plot_confusion_matrix(result, [str(classes[i]) for i in classes], normalize=True, title = title, ax = ax, cmap=cmap)
    
    def update_status(self, print_out = False, logs = {}):
        description = "epoch "+str(min(self.epoch+1, self.epochs))+"/"+str(self.epochs)
        if np.shape(self.X_test)[-1] > 0 and len(self.worst_values) > 0:
            description += " -- worst acc/class: {0:.3f}".format(self.worst_values[-1])

        if self.best_result["unique"] > -1:
            description = description + " current best: "+str(float(int(self.best_result["unique"] * 10000)) / 100) + "%"
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
        
        #fig = plt.figure(figsize=(25,10))
        #gs = fig.add_gridspec(2,2, width_ratios=[1,3])
        #ax = fig.add_subplot(gs[0,0])
        #dx = fig.add_subplot(gs[0,1])
        #bx = fig.add_subplot(gs[1,:])
        
        self.epoch = min(epoch + 1, self.epochs)

        if np.shape(self.X_test)[-1] > 0:
            result, predictions = self.plot_comparison_raw(do_plot = False, length = -1)
            
            #dx.plot(np.arange(len(result)), result[:, 3], label="% correct")
            #dx.scatter(np.arange(len(result)), result[:, 3])
            #for i in np.arange(len(result)):
            #    dx.annotate(str(classes[i])+": "+str(int(result[:, 3][i]*100))+"%", (i + 0.01, result[:, 3][i] + 0.01))
            #dx.set_ylim(0, 1)
            #dx.set_xticks(np.arange(len(result)))
            #dx.set_xticklabels([str(classes[i]) for i in classes])
            #plt.title("epoch "+str(epoch)+" range "+str(np.array(self.settings["global_segment"])))
            #utils.show_figure("classes recognition")
        
            #self.plot_confusion(np.concatenate(self.X_test, axis=0), np.concatenate(self.Y_test, axis=0), predictions = predictions, ax = ax)

            for i in range(0, len(result[:, 3])):
                if not i in self.per_class_accuracy:
                    self.per_class_accuracy[i] = []
                self.per_class_accuracy[i].append(result[:, 3][i])
            self.mean_values.append(np.mean(result[:, 3]))
            self.worst_values.append(np.min(result[:, 3]))

            set_per_class_accuracy(result[:, 3].astype(np.float))

            #bx.fill_between(np.arange(len(self.worst_values)), self.worst_values, self.mean_values, where=np.array(self.worst_values) <= np.array(self.mean_values), facecolor="red", interpolate = True, alpha= 0.25)
            
            #bx.plot(np.arange(len(self.worst_values)), self.worst_values, label="min", color="blue")
            #bx.plot(np.arange(len(self.mean_values)), self.mean_values, label="mean")

            worst_acc_per_class = np.min(result[:, 3])
        else:
            result = None
            worst_acc_per_class = -1
        
        unique = self.estimate_uniqueness()

        if not save:
            return unique

        self.uniquenesses.append(unique)
        set_uniqueness_history(self.uniquenesses)
        #bx.plot(np.arange(len(self.uniquenesses)), self.uniquenesses, label="uniqueness", color="red")

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
                TRex.log("\t(saving weights as "+output_path+"_progress.npz)")
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
            TRex.log("-- worst_value "+str(self.worst_values[-2:]))
            if not self.model.stop_training and len(self.worst_values) >= 2 and self.settings["accumulation_step"] >= -1:
                acc = np.array(self.worst_values[-2:]) #logs[akey][-2:]
                if (acc > 0.97).all() or worst_acc_per_class >= 0.99:
                    TRex.log("[STOP] "+str(acc)+" in "+akey+" has been > 0.97 for consecutive epochs. terminating.")
                    set_stop_reason(akey+" good enough ("+str(acc)+")")
                    self.model.stop_training = True

            # check whether we are plateauing at a certain uniqueness level for a long time
            if not self.model.stop_training and len(self.uniquenesses) >= 10 and self.settings["accumulation_step"] > 0:
                acc = np.diff(self.uniquenesses[-10:]).mean() #logs[akey][-2:]
                TRex.log("Uniqueness plateau check:"+str(np.diff(self.uniquenesses[-10:]))+" -> "+str(acc))
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
                        if self.settings["accumulation_step"] > 0:
                            self.model.stop_training = True
                            set_stop_reason("small loss in consecutive epochs")
                            TRex.log("[STOP] Loss is very small in consecutive epochs (epoch "+str(epoch)+"). stopping. loss was "+str(current_loss)+" vs. "+str(mu)+" "+str(self.loss_diffs[-2:]))
                        else:
                            TRex.log("(skipping small loss stopping criterion in first accumulation step)")
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
                    if count >= 4:
                        # we seem to have started overfitting
                        set_stop_reason("overfitting")
                        TRex.log("[STOP] overfitting. stopping with loss diffs: "+str(change))
                        self.model.stop_training = True
        
        #if worst_acc_per_class >= min_acceptable_value and (worst_acc_per_class >= 0.99 or (len(self.worst_values) >= min_number_iterations and self.worst_values[-1] >= min_acceptable_value)):
         #   TRex.log("[STOP] stopping training with minimum validation accuracy of "+str(worst_acc_per_class)+" per class")
          #  self.model.stop_training = True
#else:
#            TRex.log("\tcontinuing with worst accuracy of "+str(worst_value))
        
        #if len(self.better_values) > 0:
        #    bx.scatter(np.array(self.better_values).T[0], np.array(self.better_values).T[1], label="improvements", color="red")
        #    for i,j in zip(np.array(self.better_values).T[0], np.array(self.better_values).T[1]):
        #        bx.annotate(str(int(j * 1000) / 10.0)+"%",xy=(i,j), xytext=(1,5), textcoords='offset points')
        #if len(self.much_better_values) > 0:
        #    bx.scatter(np.array(self.much_better_values).T[0], np.array(self.much_better_values).T[1], label="eligble", color="green")
        
        #if self.compare_acc > 0:
        #    bx.axhline(y=self.compare_acc, label="previous best", color="gray", linestyle="--")
        #    bx.axhline(y=self.compare_acc**2, label="previous accept.", color="gray", linestyle="--")
        #bx.set_ylim(-0.05, 1.05)
        
        #if len(self.uniquenesses) < 20:
        #    bx.set_xticks(np.arange(len(self.uniquenesses)))
        #else:
        #    bx.set_xticks(np.arange(len(self.uniquenesses), 5))
        #bx.set_xlabel("epochs")
        #bx.set_ylabel("accuracy")
        #plt.legend()
        #plt.close(plt.gcf())
        #im = utils.figure_as_image()
        #TRex.imshow("training overview", im)
        #utils.show_figure(self.filename + self.prefix + " " + str(epoch) + " training overview", output_path=self.settings["output_path"], im = im)
        self.update_status(True, logs=logs)
        self.batches = 0
        return unique
    
    def on_epoch_end(self, epoch, logs={}):
        worst_value = self.evaluate(epoch, True, logs)

        global gui_terminated
        if gui_terminated():
            global UserCancelException
            self.model.stop_training = True
            #TRex.log("aborting because we have been asked to by main")
            raise UserCancelException()
    def on_batch_end(self, batch, logs={}):
        global gui_terminated
        if gui_terminated():
            global UserCancelException
            self.model.stop_training = True
            raise UserCancelException()
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

def predict():
    global receive, images, model
    
    train_X = np.array(images, copy=False)
    if len(train_X.shape) != 4:
        print("error with the shape")
        
    indexes = np.array(np.arange(len(train_X)), dtype=np.float32)
    output = np.array(model.predict(train_X), dtype=np.float32)
    
    receive(output, indexes)

    del output
    del train_X
    del indexes
    del images


def start_learning():
    global best_accuracy_worst_class, max_epochs, image_width, image_height
    global output_path, classes, learning_rate, accumulation_step, global_segment
    global batch_size, X_val, Y_val, X, Y, run_training, save_weights_after, do_save_training_images

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
        "min_iterations": 2
        #"min_acceptable_value": 0.98
    }

    X_test = np.array(X_val, copy=False) #True, dtype = float) / 255.0
    Y_test = np.array(Y_val, dtype=float)
    original_classes_test = np.copy(Y_test)
    Y_test = np_utils.to_categorical(Y_test, len(classes))

    X_train = np.array(X, copy=False)
    Y_train = np.array(Y, dtype=float)

    original_classes = np.copy(Y_train)
    #test_original_classes = np.copy(test_Y)
    Y_train = np_utils.to_categorical(Y_train, len(classes))
    #Y_test = np_utils.to_categorical(test_Y, len(classes))

    TRex.log("Python received "+str(X_train.shape)+" training images ("+str(Y_train.shape)+") and  "+str(X_test.shape)+" validation images ("+str(Y_test.shape)+")")

    mi = 0
    for i, c in zip(np.arange(len(classes)), classes):
        mi = max(mi, len(Y_train[np.argmax(Y_train, axis=1) == c]))

    plot_medians(filename + output_prefix + " dataset", X_train, np.argmax(Y_train, axis=1), np.array(classes, dtype=int))
    per_epoch = max(settings["min_iterations"], int(len(X_train) // batch_size ) * 0.5 ) #* 2.0) # i am using augmentation
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
                                 #rotation_range = 360,
                                 #brightness_range=(0.5,1.5),
                                 width_shift_range=move_range,
                                 height_shift_range=move_range,
                                 cval = 0,
                                 fill_mode = "constant",
                                 #preprocessing_function=preprocess,
                                 #use_multiprocessing=True
                                 )

    cvalues = np.array(cvalues)

    TRex.log("# [init] weights per class "+str(per_class))
    TRex.log("# [training] data shapes: train "+str(X_train.shape)+" "+str(Y_train.shape)+" test "+str(Y_test.shape)+" "+str(classes))

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
            
            validation_data = (X_test, Y_test)
            if len(X_test) == 0:
                validation_data = None

            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                          validation_data=validation_data,
                                          steps_per_epoch=per_epoch, epochs=epochs,
                                          callbacks = [ callback ],
                                          #class_weight = per_class
                                          )
            
            model_json = model.to_json()
            with open(output_path+".json", "w") as f:
                f.write(model_json)
            
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
            else:
                TRex.warn("could not improve upon previous steps.")
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
