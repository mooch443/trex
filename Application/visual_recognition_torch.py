from fileinput import filename
import re
from tkinter import NO
from turtle import clear
from matplotlib.pylab import f
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import TRex
from visual_identification_network_torch import ModelFetcher

static_inputs : torch.Tensor = None
static_targets : torch.Tensor = None

class UserCancelException(Exception):
    """Raised when user clicks cancel"""
    pass

class UserSkipException(Exception):
    """Raised when user clicks cancel"""
    pass

'''

# Mock implementations for the C++ functions
def update_work_description(description):
    print(f"Updating work description: {description}")

def set_stop_reason(reason):
    print(f"Setting stop reason: {reason}")

def set_per_class_accuracy(accuracy):
    print(f"Setting per class accuracy: {accuracy}")

def set_uniqueness_history(uniquenesses):
    print(f"Setting uniqueness history: {uniquenesses}")

def update_work_percent(percent):
    #print(f"Updating work percent: {percent * 100}%")
    pass

def acceptable_uniqueness():
    return 0.9

def accepted_uniqueness():
    return 0.95

def get_abort_training():
    return False

def get_skip_step():
    return False

def estimate_uniqueness():
    return np.random.rand()

class TRex:
    @staticmethod
    def log(message):
        print(f"TRex log: {message}")

    @staticmethod
    def warn(message):
        print(f"TRex warn: {message}")
'''
'''
def load_mnist_data(N = 5000):
    # Load MNIST dataset
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0) * 255)  # Convert to channels last
    ])

    train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=mnist_transform)
    val_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=mnist_transform)

    indexes = np.arange(len(train_dataset))
    np.random.shuffle(indexes)

    # Extract a fixed set of training samples
    X_train = []
    Y_train = []
    for i in range(N):  # Taking 500 samples from MNIST training data
        X_train.append(train_dataset[indexes[i]][0].numpy())
        Y_train.append(train_dataset[indexes[i]][1])

    print(Y_train)
    import matplotlib.pyplot as plt
    plt.imshow(X_train[0].reshape(28, 28), vmin=0, vmax=255, cmap='gray')
    plt.show()

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Extract a fixed set of validation samples
    indexes = np.arange(len(val_dataset))
    np.random.shuffle(indexes)

    X_val = []
    Y_val = []
    for i in range(N // 4):  # Taking 100 samples from MNIST validation data
        X_val.append(val_dataset[indexes[i]][0].numpy())
        Y_val.append(val_dataset[indexes[i]][1])
    
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    plt.imshow(X_val[0].reshape(28, 28), vmin=0, vmax=255, cmap='gray')
    plt.show()

    return X_train, Y_train, X_val, Y_val

#X, Y, X_val, Y_val = load_mnist_data()

def load_cifar10_data(N = 50000):
    # Load CIFAR-10 dataset
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0) * 255)  # Convert to channels last
    ])

    train_dataset = datasets.CIFAR10(root='cifar10_data', train=True, download=True, transform=cifar_transform)
    val_dataset = datasets.CIFAR10(root='cifar10_data', train=False, download=True, transform=cifar_transform)

    indexes = np.arange(len(train_dataset))
    np.random.shuffle(indexes)

    N = min(N, len(train_dataset))

    # Extract a fixed set of training samples
    X_train = []
    Y_train = []
    for i in range(N):  # Taking 500 samples from CIFAR-10 training data
        X_train.append(train_dataset[indexes[i]][0].numpy())
        Y_train.append(train_dataset[indexes[i]][1])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Extract a fixed set of validation samples
    indexes = np.arange(len(val_dataset))
    np.random.shuffle(indexes)

    X_val = []
    Y_val = []
    for i in range(min(N // 4, len(val_dataset))):  # Taking 100 samples from CIFAR-10 validation data
        X_val.append(val_dataset[indexes[i]][0].numpy())
        Y_val.append(val_dataset[indexes[i]][1])
    
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    return X_train, Y_train, X_val, Y_val

X, Y, X_val, Y_val = load_cifar10_data(N=150000)

image_channels = X.shape[3]
image_width = X.shape[1]
image_height = X.shape[2]
#image_channels = X.shape[1]
#image_width = X.shape[2]
#image_height = X.shape[3]

best_accuracy_worst_class = 0.8
max_epochs = 150
output_path = "output"
classes = np.unique(Y)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
learning_rate = 0.001
accumulation_step = 1
global_tracklet = list(range(5))
verbosity = 1
batch_size = 128

TRex.log(f"Loaded MNIST data with shape: {image_width}x{image_height}x{image_channels}")
TRex.log(f"Classes: {classes}")

#X_val = np.random.rand(100, image_height, image_width, 1)
#Y_val = np.random.randint(0, 10, 100)
#X = np.random.rand(500, image_height, image_width, 1)
#Y = np.random.randint(0, 10, 500)
run_training = True
save_weights_after = True
do_save_training_images = lambda: False
min_iterations = 10
filename = "model"
output_prefix = "output"

#batch_size = 16#int(max(batch_size, 64))
epochs = max_epochs
'''

'''run_training = None
image_channels = None
image_width : int = None
image_height : int = None
output_path = None
classes = None
learning_rate = None
accumulation_step = None
global_tracklet = None
verbosity = None
best_accuracy_worst_class = None
do_save_training_images = None
output_prefix = None
filename = None

save_weights_after = None

X = None
Y = None
X_val = None
Y_val = None

model = None
device = 'mps'

min_iterations = None
max_epochs = None
batch_size = None

network_version = None'''

# alternative for onehotencoder from sklearn:
class OneHotEncoder:
    def __init__(self, sparse_output = False):
        self.categories_ = None

    def fit(self, y):
        self.categories_ = np.unique(y).tolist()
        return self

    def transform(self, y : np.ndarray):
        if self.categories_ is None:
            raise RuntimeError("You must fit the encoder before transforming data.")

        num_classes = len(self.categories_)
        one_hot = np.zeros((len(y), num_classes))
        for i, c in enumerate(y):
            one_hot[i, self.categories_.index(c)] = 1
        return one_hot

    def fit_transform(self, y : np.ndarray):
        return self.fit(y).transform(y)

class CustomDataLoader:
    def __init__(self, X : np.ndarray, Y : np.ndarray, batch_size : int, shuffle : bool = False, transform : transforms.Compose = None, device : str = 'cpu'):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.device = device

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self.indexes = np.arange(len(self.X))
            np.random.shuffle(self.indexes)
        return self

    def __next__(self) -> tuple[torch.Tensor, np.ndarray]:
        if self.index >= len(self.X):
            raise StopIteration
        elif not self.shuffle:
            if self.transform is None:
                x = torch.tensor(self.X[self.index:self.index+self.batch_size], dtype=torch.float32, device=self.device)
                y = self.Y[self.index:self.index+self.batch_size]
            else:
                x = self.transform(torch.tensor(self.X[self.index:self.index+self.batch_size], dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255).permute(0, 2, 3, 1) * 255
                y = self.Y[self.index:self.index+self.batch_size]
            self.index += self.batch_size
            return (x, y)
        else:
            indexes = self.indexes[self.index:self.index+self.batch_size]
            if self.transform is not None:
                x = self.transform(torch.tensor(self.X[indexes], dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255).permute(0, 2, 3, 1) * 255
                y = self.Y[indexes]
            else:
                x = torch.tensor(self.X[indexes], dtype=torch.float32, device=self.device)
                y = self.Y[indexes]
            self.index += self.batch_size
            return (x, y)

    def __len__(self):
        return len(self.X) // self.batch_size

def clear_caches():
    TRex.log("Clearing caches...")
    #if device == 'cuda':
    #    TRex.log(f"Before clearing: {torch.cuda.memory_summary()}")
    #elif device == 'mps':
    #    TRex.log(f"Before: {X} {Y} {X_val} {Y_val} {model}")

    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        current_mem=torch.mps.current_allocated_memory()
        torch.mps.empty_cache()
        TRex.log(f"Current memory: {current_mem/1024/1024}MB -> {torch.mps.current_allocated_memory()/1024/1024}MB")

    import gc
    gc.collect()

p_softmax = None

def predict_numpy(model, images, batch_size, device):
    global p_softmax

    model.eval()
    #assert model.device == device

    # predict in batches
    output = []
    with torch.no_grad():
        if p_softmax is None:
            p_softmax = nn.Softmax(dim=1).to(device)
        for i in range(0, len(images), batch_size):
            x = torch.tensor(images[i:i+batch_size], dtype=torch.float32, requires_grad=False).to(device).detach()
            output.append(p_softmax(model(x)).cpu().numpy())
            del x
        #del softmax

    #return torch.cat(output, dim=0)
    return np.concatenate(output, axis=0)


class ValidationCallback:
    def __init__(self, model : nn.Module, classes : list, X_test : np.ndarray, Y_test : np.ndarray, epochs : int, filename : str, prefix : str, output_path : str, compare_acc : float, settings : object, device : str):
        self.classes = classes
        self.model = model
        self.prefix = prefix
        self.output_path = output_path
        self.compare_acc = compare_acc
        self.device = device
        
        X = []
        Y = []
        if len(X_test) > 0:
            for c in classes:
                #mask = Y_test.argmax(axis=1).eq(c)
                mask = Y_test.argmax(axis=1) == c
                a = X_test[mask]
                X.append(a)
                Y.append(Y_test[mask])
        
        self.X_test : list[np.ndarray] = X
        self.Y_test : list[torch.Tensor] = Y
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
        self.stop_training = False
        
    def plot_comparison_raw(self):
        classes = self.classes

        result = np.zeros((len(classes), 4), dtype=float)
        
        self.model.eval()

        # compute a matrix with the median class, the standard deviation of the distance from Y to the correct 1-hot vector, the median sum of all classes and the accuracy
        for i, c, images, should in zip(np.arange(len(classes)), classes, self.X_test, self.Y_test):
            if len(images) == 0:
                continue

            # predict in batches
            with torch.no_grad():
                #should = should.cpu().numpy()#clone().detach().type(torch.float32).to(self.device)
                y = predict_numpy(self.model, images, self.settings["batch_size"], device=self.device)#.cpu()
            #TRex.log(f"Class {c} - {len(images)} images: {y.shape} - {should.shape} {y.dtype} {should.dtype}")
            
            #TRex.log(f"Class {c} - {len(images)} images: {y.shape} - {should.shape} {y.dtype} {should.dtype}")
            #TRex.log(f"Y: {y} - should: {should}")
            #TRex.log(f"{y - should}")

            # shape of y: (1, num_classes)
            #TRex.log(f"y.shape = {y.shape}, should.shape = {should.shape}")
            #distance : torch.Tensor = torch.abs(y - should).sum(dim=1)
            distance : np.ndarray = np.abs(y - should).sum(axis=1)
            #TRex.log(f"Distance: {distance}")
            #TRex.log(f"Values: {np.abs(y - should)}")

            # calculate the median class
            #result[i, 0] = torch.median(y.argmax(dim=1)).item()
            result[i, 0] = np.median(y.argmax(axis=1))

            # calculate the standard deviation of the distance to the correct class
            #result[i, 1] = distance.std().item()
            result[i, 1] = distance.std()

            # calculate the median sum of all classes
            #result[i, 2] = torch.sum(y, dim=1).median().item()
            result[i, 2] = np.median(np.sum(y, axis=1))

            # calculate the accuracy of the class
            #result[i, 3] = (y.argmax(dim=1) == i).float().sum().item() / len(y)
            result[i, 3] = (y.argmax(axis=1) == i).sum() / len(y)

            #result[i, 0] = np.median(np.argmin(distance, axis=1))
            #result[i, 1] = np.std(np.min(distance, axis=1))
            #result[i, 2] = np.median(np.sum(Y, axis=1))
            #result[i, 3] = np.mean(np.argmax(Y, axis=1) == np.argmax(should, axis=1))

            del images
            del should
            del y
        
        return result
    
    def update_status(self, print_out=False, logs={}, patience=5):
        global update_work_description

        description = f"Epoch <c><nr>{min(self.epoch+1, self.epochs)}</nr></c>/<c><nr>{self.epochs}</nr></c>"
        #if len(self.X_test) > 0 and len(self.worst_values) > 0:
        #    description += f" -- worst acc/class: <c><nr>{(self.worst_values[-1]*100):.1f}</nr></c><i>%</i>"
        description += " --"

        if self.best_result["unique"] > -1:
            if self.compare_acc <= 0 or self.best_result["unique"] >= self.compare_acc:
                description += f" best uniqueness: <c><nr>{(self.best_result['unique']*100):.2f}</nr></c><i>%</i>"
                if self.compare_acc > 0:
                    description += f" <gray>(<lightgray>prev.: <c><nr>{(self.compare_acc*100):.2f}</nr></c><i>%</i></lightgray>)</gray>"
            else:
                description += f" <gray>best uniqueness: <c><nr>{(self.best_result['unique']*100):.2f}</nr></c><i>%</i></gray>"
                description += f" (<lightgray>prev.</lightgray>: <c><nr>{(self.compare_acc*100):.2f}</nr></c><i>%</i>)"
            
        elif self.compare_acc > 0:
            description += f" <lightgray>previous best uniqueness</lightgray>: <c><nr>{(self.compare_acc*100):.2f}</nr></c><i>%</i>"
        else:
            description += f" <lightgray>no recorded uniqueness</lightgray>"
        if len(self.losses) >= patience:
            description += f" <sym>△</sym>loss: <c><nr>{(abs(np.mean(self.loss_diffs[-5:])) / self.minimal_loss_allowed*100):.2f}</nr></c><i>%</i> of minimum"
        
        update_work_description(description)
        description = f"[TRAIN] {description}"

        if print_out:
            TRex.log(f"{description} {str(logs)}")

    def evaluate(self, epoch, save=True, logs={}):
        global update_work_percent, set_stop_reason, set_per_class_accuracy, set_uniqueness_history, estimate_uniqueness

        classes = self.classes
        model = self.model
        patience = 8
        
        update_work_percent(min(epoch + 1, self.epochs) / self.epochs)
        
        self.epoch = min(epoch + 1, self.epochs)

        if len(self.X_test) > 0:
            TRex.log(f"Comparing epoch {epoch} with {len(self.X_test)} classes")
            result = self.plot_comparison_raw()
            TRex.log(f"Result: {result[:, 3]}")

            '''import matplotlib.pyplot as plt
            plt.plot(result[:, 3])
            plt.ylim(0, 1)
            plt.ylabel('Accuracy / Class')
            plt.xlabel('Class')
            plt.show()'''
            
            for i in range(len(result[:, 3])):
                if i not in self.per_class_accuracy:
                    self.per_class_accuracy[i] = []
                self.per_class_accuracy[i].append(result[:, 3][i])
            self.mean_values.append(np.mean(result[:, 3]))
            self.worst_values.append(np.min(result[:, 3]))

            set_per_class_accuracy(result[:, 3].astype(float))
            worst_acc_per_class = np.min(result[:, 3])
        else:
            result = None
            worst_acc_per_class = -1
        
        TRex.log(f"Estimating uniqueness for epoch {epoch}...")
        unique = estimate_uniqueness()
        TRex.log(f"Uniqueness: {unique}")

        if not save:
            return unique

        self.uniquenesses.append(unique)
        set_uniqueness_history(self.uniquenesses)

        pessimism = (self.compare_acc * 0.95)**2

        if unique >= acceptable_uniqueness() and self.settings["accumulation_step"] >= -1:
            if self.settings["accumulation_step"] == -1:
                self.stop_training = True
                set_stop_reason(f"Uniqueness is sufficient ({unique:.2f})")
            elif unique >= accepted_uniqueness():
                self.stop_training = True
                set_stop_reason(f"Uniqueness is sufficient ({unique:.2f})")

        if unique > self.best_result["unique"] and (self.compare_acc <= 0 or unique >= pessimism):
            self.best_result["unique"] = unique
            self.better_values.append((epoch, unique))
            TRex.log(f"\t(saving) new best-worst-value: {unique}")
            if unique >= self.compare_acc:
                self.much_better_values.append((epoch, unique))

            global output_path
            self.best_result["weights"] = model.state_dict()
            try:
                TRex.log(f"\t(saving model as '{output_path}_progress.pth')")
                torch.save(model.state_dict(), output_path+"_progress.pth")
            except Exception as e:
                TRex.warn(str(e))
        else:
            TRex.log(f"\t(not saving) old best value is {self.best_result['unique']} / {unique}")

        if self.compare_acc > 0 and len(self.uniquenesses) >= patience and self.best_result["unique"] < pessimism:
            set_stop_reason(f"uniqueness stayed below {int(pessimism * 10000.0) / 100.0}% in the first epochs")
            TRex.log(f"[STOP] best result is below {pessimism} even after {epoch} epochs.")
            self.stop_training = True

        key = "val_loss" if "val_loss" in logs else "loss"
        akey = "val_acc" if "val_acc" in logs else "acc"

        assert key in logs
        current_loss = logs[key]
        previous_loss = np.finfo(float).max
        
        if len(self.losses) > 0:
            l = np.nanmean(self.losses[-patience:])
            if not np.isnan(l):
                previous_loss = l
        self.losses.append(current_loss)
        if len(self.losses) > 1:
            mu = np.mean(self.losses[-patience:-1])
            self.loss_diffs.append((current_loss - mu))

        loss_diff = max(0.00000001, previous_loss - current_loss)
        self.minimal_loss_allowed = 0.05 * 10**(int(np.log10(current_loss)) - 1)
        change = np.diff(self.losses)

        if len(self.losses) > 1:
            TRex.log(f"\tminimal_loss_allowed: {self.minimal_loss_allowed} -> mu: {mu} current diffs: {self.loss_diffs} average: {np.mean(self.loss_diffs[-patience:])}")

        if not self.stop_training:
            long_time = int(max(8, self.epochs * 0.1))
            long_time = min(long_time, 13)

            worst_value_backlog = patience
            TRex.log(f"-- worst_value (backlog={worst_value_backlog}) {self.worst_values[-worst_value_backlog:]} -- long time: {long_time}")
            if not self.stop_training and len(self.worst_values) >= worst_value_backlog and self.settings["accumulation_step"] >= -1:
                acc = np.array(self.worst_values[-worst_value_backlog:])
                if (acc > 0.97).all() or worst_acc_per_class >= 0.99:
                    TRex.log(f"[STOP] {acc} in {akey} has been > 0.97 for consecutive epochs. terminating.")

                    rounded_str = ", ".join([f'{x:.2f}' for x in acc])
                    set_stop_reason(f"{akey} good enough ({rounded_str})")
                    self.stop_training = True

            if not self.stop_training and len(self.uniquenesses) >= long_time and self.settings["accumulation_step"] > 0:
                acc = np.diff(self.uniquenesses[-long_time:]).mean()
                TRex.log(f"Uniqueness plateau check: {np.diff(self.uniquenesses[-long_time:])} -> {acc}")
                if acc <= 0.01:
                    set_stop_reason("uniqueness plateau")
                    rounded_str = ", ".join([f'{x:.2f}' for x in self.uniquenesses[-long_time:]])
                    TRex.log(f"[STOP] Uniqueness has been plateauing for several epochs. terminating: {rounded_str}")
                    self.stop_training = True

            if not self.stop_training and len(self.losses) > 1:
                if len(self.losses) >= patience and abs(np.mean(self.loss_diffs[-patience:])) < self.minimal_loss_allowed:
                    if self.settings["accumulation_step"] > 0 or (self.last_skip_step == self.settings["accumulation_step"] and len(self.uniquenesses) >= self.epochs * 0.5):
                        self.stop_training = True
                        set_stop_reason("small loss in consecutive epochs")
                        if self.settings["accumulation_step"] <= 0:
                            TRex.log(f"[STOP] Before second accumulation step, but #{len(self.uniquenesses)} epochs / {self.epochs} is more than 50%.")
                        else:
                            TRex.log(f"[STOP] Loss is very small in consecutive epochs (epoch {epoch}). stopping. loss was {current_loss} vs. {mu} {self.loss_diffs[-2:]}")
                    else:
                        TRex.log("(skipping small loss stopping criterion in first accumulation step)")
                        self.last_skip_step = self.settings["accumulation_step"]
                elif len(change) >= 2:
                    TRex.log(f"\t{current_loss} => {current_loss - mu} ({current_loss} / {mu})")
                count = 0
                for i in range(-10, -1):
                    v = change[i:i+1]
                    if len(v) > 0 and v[0] >= 0:
                        count += 1
                    elif len(v) > 0:
                        count = 0
                TRex.log(f"\tcounted {count} increases in loss in consecutive epochs - {change}")
                if count >= 4 and (self.settings["accumulation_step"] != 0 or (self.settings["accumulation_step"] == 0 and len(self.losses) >= 30)):
                    set_stop_reason("overfitting")
                    TRex.log(f"[STOP] overfitting. stopping with loss diffs: {change}")
                    self.stop_training = True

        self.update_status(True, logs=logs, patience=patience)
        self.batches = 0
        return unique
    
    def on_epoch_end(self, epoch, logs={}):
        TRex.log(f"Epoch {epoch}/{self.epochs} ended: {logs}")
        worst_value = self.evaluate(epoch, True, logs)

        global get_abort_training, get_skip_step
        if get_abort_training():
            global UserCancelException
            self.stop_training = True
            raise UserCancelException()
        if get_skip_step():
            global UserSkipException
            self.stop_training = True
            raise UserSkipException()

    def on_batch_end(self, batch, logs={}):
        #TRex.log(f"Batch {batch} ended")
        global get_abort_training, get_skip_step
        if get_abort_training():
            global UserCancelException
            self.stop_training = True
            raise UserCancelException()
        if get_skip_step():
            global UserSkipException
            self.stop_training = True
            raise UserSkipException()

        self.batches += 1
        if batch % 50 == 0:
            self.update_status(logs=logs)
            global update_work_percent
        epoch = self.epoch
        epoch += self.batches / self.settings["per_epoch"]
        #TRex.log(f"Epoch: {epoch}: {self.batches} / {self.settings['per_epoch']}")
        update_work_percent(epoch / self.epochs)


def reinitialize_network():
    global image_channels, device
    global model, image_width, image_height, classes, learning_rate, sess, network_version

    # if no device is specified, use cuda if available, otherwise use mps/cpu
    device = TRex.choose_device()
    assert device is not None

    model = ModelFetcher().get_model(network_version, len(classes), image_channels, image_width, image_height, device=device)
    TRex.log(f"Reinitialized network with version {network_version}:\n{model}")
    TRex.log(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    TRex.log(f"Device: {device}")

def load_weights():
    global output_path, model, accumulation_step
    import os
    try:
        if os.path.exists(output_path+"_model.pth"):
            TRex.log(f"Loading entire model from {output_path+'_model.pth'} in step {accumulation_step}")
            model = torch.load(output_path+'_model.pth', weights_only=False)
        else:
            TRex.log(f"Loading weights from {output_path+'.pth'} in step {accumulation_step}")
            model.load_state_dict(torch.load(output_path+'.pth', weights_only=True))
        
    except Exception as e:
        TRex.warn(str(e))
        raise Exception("Failed to load weights. This is likely because either the individual_image_size is different, or because existing weights might have been generated by a different version of TRex (see visual_identification_version).")

def predict():
    global receive, images, model, image_channels, device, batch_size, image_width, image_height
    images = np.array(images, copy=False)
    TRex.log(f"Predicting {len(images)} images with shape {images.shape}")
    
    if len(images.shape) != 4:
        TRex.warn(f"error with the shape {images.shape} < len 4")
    if images.shape[3] != image_channels:
        TRex.warn(f"error with the shape {images.shape} < channels {image_channels}")
    if images.shape[1] != image_width or images.shape[2] != image_height:
        TRex.warn(f"error with the shape {images.shape} < width {image_width} height {image_height}")
    #elif train_X.shape[1] != model.input.shape[1] or train_X.shape[2] != model.input.shape[2]:
    #    raise Exception("Wrong image dimensions for model ("+str(train_X.shape[1:3])+" vs. "+str(model.input.shape[1:3])+").")
        
    indexes = np.array(np.arange(len(images)), dtype=np.float32)
    #output = model.predict(tf.cast(train_X, dtype=np.float32), verbose=0)

    output = predict_numpy(model, images, batch_size, device).astype(np.float32)#.cpu().numpy()
    TRex.log(f"Predicted images with shape {output.shape}")
    
    receive(output, indexes)

    del output
    del indexes
    del images

    import gc
    gc.collect()

def train(model, train_loader, val_loader, criterion, optimizer : torch.optim.Adam, callback : ValidationCallback, scheduler, transform, settings, device = 'mps'):
    global get_abort_training
    global static_inputs, static_targets
    num_classes = len(settings["classes"])

    import matplotlib.pyplot as plt
    import torchmetrics
    import gc
    from tqdm import tqdm

    # Initialize metrics
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)

    best_val_acc = 0.0
    softmax = nn.Softmax(dim=1)#.to(device)
    
    if static_inputs is None or static_inputs.shape[-1] != settings["image_channels"]:
        static_inputs = torch.empty((settings["batch_size"], settings["image_height"], settings["image_width"], settings["image_channels"]), dtype=torch.float32, device=device, requires_grad=False).detach()
        static_targets = torch.empty((settings["batch_size"], num_classes), dtype=torch.float32, device='cpu', requires_grad=False).detach()

    #import tracemalloc;
    #tracemalloc.start(50)

    last_snapshot = None
    
    for epoch in range(settings["epochs"]):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        #snapshot0 = tracemalloc.take_snapshot()
        #import torch.autograd.profiler as profiler

        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        if True:
            # dont use train_loader, manually iterate over the dataset
            for batch, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                #inputs = transform(inputs.permute(0, 3, 1, 2) / 255).permute(0, 2, 3, 1) * 255

                #inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
                targets = torch.tensor(targets, dtype=torch.float32)#, device=device)

                #for i in range(0, len(X), settings["batch_size"]):
                #batch = i // settings["batch_size"]

                #inputs = torch.tensor(X[i:i+settings["batch_size"]], dtype=torch.float32, device=device)
                #targets = torch.tensor(callback.Y_one_hot[i:i+settings["batch_size"]], dtype=torch.float32, device=device)

                #TRex.log(f"Batch {batch}/{len(X)} - inputs: {inputs.shape} targets: {targets.shape}")
                
                #for batch, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                

                '''if batch == 0 and epoch == 0:
                    print(f"Batch {batch}/{len(train_loader)} - inputs: {inputs.shape} targets: {targets.shape} - {torch.argmax(targets, dim=1)[0]}")

                    import torchvision
                    grid = torchvision.utils.make_grid(inputs.permute(0, 3, 1, 2), nrow=16, value_range=(0,255))
                    image = grid.permute(1, 2, 0).numpy().astype(np.uint8)

                    TRex.log(f"Image shape: {image.shape}")
                    TRex.imshow("batch0", image)
                    #plt.imshow(grid.permute(1, 2, 0).numpy().astype(int))
                    #plt.show()

                    #return'''
                #print(f"Batch {batch}/{len(train_loader)} - inputs: {inputs.shape} targets: {targets.shape} - {torch.argmax(targets, dim=1)[0]}")
                #static_inputs = inputs.clone()
                #static_targets = targets.clone()
                static_inputs.resize_(inputs.shape).copy_(inputs)
                static_targets.resize_(targets.shape).copy_(targets)

                #assert static_inputs.is_contiguous()
                #assert static_targets.is_contiguous()

                #print(f"Device: {static_inputs.device} vs. {device} vs. {inputs.device}")
                #assert static_inputs.device == device
            
                outputs = model(static_inputs).cpu()
                loss = criterion(outputs, static_targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    outputs = softmax(outputs.detach())
                    acc = torch.mean((outputs.argmax(dim=1) == static_targets.argmax(dim=1)).float())
                
                running_acc += acc
                running_loss += loss.item()

                logs = {'loss': loss.item(), 'acc': acc}
                callback.on_batch_end(batch, logs)

                del inputs
                del targets
                del outputs
                del loss
                del acc
                del logs

                #gc.collect()

        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        #gc.collect()

        running_loss /= len(train_loader)
        acc = running_acc / len(train_loader)

        # Check gradients (debugging)
        '''for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'{name} - Gradients: {param.grad.norm()}')
            else:
                print(f'{name} - No gradients')'''

        #gc.collect()
        #snapshot2 = tracemalloc.take_snapshot()

        # print stack traces with the most allocations
        '''traces = snapshot2.compare_to(snapshot0, 'traceback')[:5]
        for index, stat in enumerate(traces):
            TRex.log(f"Most allocated: {stat.size} bytes - {stat.count} times")
            for line in stat.traceback.format():
                TRex.log(line)

        del snapshot0
        del snapshot2
        del traces'''

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        val_loss = 0
        correct = 0
        total = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                #for inputs, targets in val_loader:
                #for batch, i in enumerate(range(0, len(X_val), settings["batch_size"])):
                #    inputs = torch.tensor(X_val[i:i+settings["batch_size"]], dtype=torch.float32, device=device).detach()
                #    targets = callback.Y_val_one_hot[i:i+settings["batch_size"]].clone().detach().to(device=device)

                for batch, (inputs, targets) in enumerate(val_loader):
                    #inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
                    targets = torch.tensor(targets, dtype=torch.float32)#, device=device)

                    static_inputs.resize_(inputs.shape).copy_(inputs)
                    static_targets.resize_(targets.shape).copy_(targets)

                    #inputs = inputs.to(device)
                    #targets = targets.to(device)

                    #outputs = model(inputs)
                    outputs = model(static_inputs).cpu().detach()
                    loss = criterion(outputs, static_targets)
                    val_loss += loss.item()

                    outputs = softmax(outputs)

                    # Calculate the accuracy of the model on the validation set
                    # by counting the number of correct predictions.
                    _, predicted = outputs.max(1)

                    total += static_targets.size(0)
                    correct += predicted.eq(static_targets.argmax(dim=1)).sum().item()

                    # Update precision and recall
                    #precision.update(predicted, static_targets.argmax(dim=1))
                    #recall.update(predicted, static_targets.argmax(dim=1))

                    del inputs
                    del targets
                    del outputs
                    del loss
                    del predicted

            val_loss /= len(val_loader)
            val_acc = correct / total
            
            # Compute precision and recall
            val_precision = 0#precision.compute().item()
            val_recall = 0#recall.compute().item()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                #torch.save(model.state_dict(), f"{settings['output_path']}_{settings['model']}_{epoch}_{val_acc:.4f}.pth")
                #TRex.log(f"Saved best model with val_acc: {best_val_acc:.4f}")
            
            # Step the scheduler
            scheduler.step(val_loss)

            logs = {'val_loss': val_loss, 'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall}
            callback.on_epoch_end(epoch, logs)

            try:
                lr = scheduler.get_last_lr()[0]
            except AttributeError:
                lr = optimizer.param_groups[0]['lr']

            TRex.log(f"Epoch {epoch}/{settings['epochs']} - loss: {running_loss} val_loss: {val_loss} - acc: {acc} val_acc: {val_acc} - precision: {val_precision} - recall: {val_recall} - LR: {lr}")

            del val_loss
            del val_acc
            del val_precision
            del val_recall
        else:
            callback.on_epoch_end(epoch, {'loss': running_loss, 'acc': acc})

            try:
                lr = scheduler.get_last_lr()[0]
            except AttributeError:
                lr = optimizer.param_groups[0]['lr']
            
            TRex.log(f"Epoch {epoch}/{settings['epochs']} - loss: {running_loss} - acc: {acc} - LR: {lr}")

        #TRex.log(f"Epoch {epoch}/{settings['epochs']}")

        if callback.stop_training:
            break
        if get_abort_training():
            break
            #raise UserCancelException()
        
        #gc.collect()
    
    #del static_inputs
    #del static_targets

    del precision
    del recall

    #tracemalloc.stop()

    clear_caches()
    clear_caches()
    clear_caches()
    

def start_learning():
    global image_channels, output_prefix, filename
    global best_accuracy_worst_class, max_epochs, image_width, image_height, update_work_percent
    global output_path, classes, learning_rate, accumulation_step, global_tracklet, verbosity
    global batch_size, X_val, Y_val, X, Y, run_training, save_weights_after, do_save_training_images, min_iterations
    global get_abort_training, model, train, device, network_version

    move_range = min(0.05, 2 / min(image_width, image_height))
    TRex.log("setting move_range as " + str(move_range))
    TRex.log(f"# [run] Input data: {len(X)} training samples, {len(X_val)} validation samples.")

    abort_with_error = False

    settings = {
        "epochs": max_epochs,
        "batch_size": batch_size,
        "move_range": move_range,
        "output_path": output_path,
        "image_width": image_width,
        "image_height": image_height,
        "image_channels": image_channels,
        "classes": classes,#np.array(classes, dtype=int),
        "learning_rate": learning_rate,
        "accumulation_step": accumulation_step,
        "global_tracklet": np.array(global_tracklet, dtype=int),
        "per_epoch": -1,
        "min_iterations": min_iterations,
        "verbosity": verbosity,
        "save_weights_after": save_weights_after
        #"min_acceptable_value": 0.98
    }

    # Data augmentation setup
    transform = transforms.Compose([
        #transforms.RandomRotation(degrees=180),
        #transforms.RandomAffine(degrees=0, translate=(move_range, move_range)),
        transforms.ColorJitter(brightness=(0.85, 1.15),
                            contrast=(0.85, 1.15),
                            saturation=(0.85, 1.15),
                            hue=(-0.05, 0.05),
                            ),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomResizedCrop((image_height, image_width), scale=(0.95, 1.05)),
    ])

    X_train : np.ndarray = np.array(X, copy=False)
    Y_train : np.ndarray = np.array(Y, dtype=int)
    X_test : np.ndarray = np.array(X_val, copy=False)
    Y_test : np.ndarray = np.array(Y_val, dtype=int)

    # Convert Y values to one-hot encoding+
    TRex.log(f"Y_test: {Y_test.shape}")

    onehot_encoder = OneHotEncoder(sparse_output=False)
    #Y_val_one_hot : np.ndarray = onehot_encoder.fit_transform(Y_test.reshape(-1, 1))
    #Y_train_one_hot : np.ndarray = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_train = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_test = onehot_encoder.fit_transform(Y_test.reshape(-1, 1))

    #Y_val_one_hot : torch.Tensor = torch.tensor(Y_val_one_hot, dtype=torch.float32, requires_grad=False)
    #Y_train_one_hot : torch.Tensor = torch.tensor(Y_train_one_hot, dtype=torch.float32, requires_grad=False)

    # Convert numpy arrays to torch tensors
    #X_test : torch.Tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=False)
    #Y_test : torch.Tensor = Y_val_one_hot
    #X_train : torch.Tensor = torch.tensor(X, dtype=torch.FloatTensor.dtype, requires_grad=False)
    #Y_train : np.ndarray = Y_train_one_hot

    # only print this if the shape is not empty:
    if X_train.shape[0] > 0:
        TRex.log(f"X_train: {X_train.shape}, Y_train: {Y_train.shape} pixel min: {X_train.min()} max: {X_train.max()} median pixel: {np.median(X_train)}")
    else:
        # generate some dummy data
        X_train = np.random.rand(1, image_height, image_width, image_channels)
        Y_train = np.random.rand(1, len(classes))
        #X_train = torch.rand(1, image_height, image_width, image_channels)
        #Y_train = torch.rand(1, len(classes))

        TRex.log(f"Generated dummy data: X_train: {X_train.shape}, Y_train: {Y_train.shape} pixel min: {X_train.min()} max: {X_train.max()} median pixel: {np.median(X_train)}")

    if X_test.shape[0] > 0:
        TRex.log(f"X_test: {X_test.shape}, Y_test: {Y_test.shape} pixel min: {X_test.min()} max: {X_test.max()} median pixel: {np.median(X_test)}")
    else:
        # generate some dummy data
        #X_test = torch.rand(1, image_height, image_width, image_channels)
        #Y_test = torch.rand(1, len(classes))
        X_test = np.random.rand(1, image_height, image_width, image_channels)
        Y_test = np.random.rand(1, len(classes))

        TRex.log(f"Generated dummy data: X_test: {X_test.shape}, Y_test: {Y_test.shape} pixel min: {X_test.min()} max: {X_test.max()} median pixel: {np.median(X_test)}")

    #train_data = TensorDataset(X_train, Y_train)
    #val_data = TensorDataset(X_test, Y_test)

    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    #val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    train_loader = CustomDataLoader(X_train, Y_train, batch_size=batch_size, shuffle=True, transform=transform, device=device)
    val_loader = CustomDataLoader(X_test, Y_test, batch_size=batch_size)

    settings["model"] = network_version
    settings["device"] = str(device)

    #if model is None:
    #    TRex.log(f"# [init] loading model {model_name} with {num_classes} classes and {image_channels} channels ({image_width}x{image_height})")
    #    model = model_fetcher.get_model(model_name, num_classes, image_channels, image_width, image_height, device=device)

    assert model is not None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    mi = 0
    for i, c in zip(np.arange(len(classes)), classes):
        #mi = max(mi, Y_train.argmax(dim=1).eq(c).sum().item())
        mi = max(mi, len(Y_train[np.argmax(Y_train, axis=1) == c]))

    per_epoch = int(len(X_train) / batch_size + 0.5)
    settings["per_epoch"] = per_epoch
    TRex.log(str(settings))

    per_class = {}
    m = 0
    cvalues = []
    for i, c in zip(np.arange(len(classes)), classes):
        # Count the number of correct samples for each class
        # and download to cpu
        #print(Y_train.argmax(dim=1).eq(c))
        #per_class[i] = Y_train.argmax(dim=1).eq(c).sum().item()
        #print(per_class[i])
        per_class[i] = len(Y_test[np.argmax(Y_test, axis=1) == c])

        cvalues.append(per_class[i])
        if per_class[i] > m:
            m = per_class[i]

    if run_training:
        callback = ValidationCallback(model, classes, X_test, Y_test, max_epochs, filename, output_prefix, output_path, best_accuracy_worst_class, settings, device)

        TRex.log(f"# [init] weights per class {per_class}")
        TRex.log(f"# [training] data shapes: train={X_train.shape} {Y_train.shape} val={X_test.shape} {Y_test.shape} classes={classes}")

        min_pixel_value = np.min(X_train)
        max_pixel_value = np.max(X_train)

        TRex.log(f"# [values] pixel values: min={min_pixel_value} max={max_pixel_value}")
        TRex.log(f"# [values] class values: min={np.min(Y_train)} max={np.max(Y_train)}")

        if do_save_training_images():
            TRex.log(f"# [training] saving training images to {output_path}_train_images.npz")
            try:
                np.savez(output_path+"_train_images.npz", 
                        X_train=X_train,
                        Y_train=Y_train, 
                        classes=classes)
            except Exception as e:
                TRex.warn("Error saving training images: " + str(e))

        # Example training call
        #try:
        train(model, train_loader, val_loader, criterion, optimizer, callback, scheduler=scheduler, device=device, transform=transform, settings=settings)

        if callback.best_result["unique"] != -1:
            weights = callback.best_result["weights"]
            history_path = output_path+"_"+str(accumulation_step)+"_history.npz"
            TRex.log("saving histories at '"+history_path+"'")

            # load best results from the current run
            try:
                TRex.log(f"Loading weights from {output_path+'_progress.pth'} in step {accumulation_step}")
                model.load_state_dict(torch.load(output_path+'_progress.pth', weights_only=True))
            except Exception as e:
                TRex.warn(str(e))
                raise Exception("Failed to load weights. This is likely because either the individual_image_size is different, or because existing weights might have been generated by a different version of TRex (see visual_identification_version).")
            
            np.savez(history_path, #history=np.array(history.history, dtype="object"), 
                     uniquenesses=callback.uniquenesses, better_values=callback.better_values, much_better_values=callback.much_better_values, worst_values=callback.worst_values, mean_values=callback.mean_values, settings=np.array(settings, dtype="object"), per_class_accuracy=np.array(callback.per_class_accuracy),
                samples_per_class=np.array(per_class, dtype="object"))

            if save_weights_after:
                TRex.log(f"# [saving] saving model to {output_path}.pth")
                try:
                    torch.save(model.state_dict(), output_path+".pth")
                    torch.save(model, output_path+"_model.pth")
                except Exception as e:
                    TRex.warn("Error saving model: " + str(e))
                    
                TRex.log("saved model to "+output_path+" with accuracy of "+str(callback.best_result["unique"]))
        
            try:
                #for i, layer in zip(range(len(model.layers)), model.layers):
                #    if i in weights:
                #        layer.set_weights(weights[i])

                if len(X_train) > 0:
                    callback.evaluate(max_epochs, False)
                    best_accuracy_worst_class = callback.best_result["unique"]
                    TRex.log(f"Best uniqueness of step {accumulation_step}: {best_accuracy_worst_class}")
            except:
                TRex.warn("loading weights failed")
            
        elif settings["accumulation_step"] == -2:
            TRex.warn("could not improve upon previous steps.")
        else:
            TRex.warn("Could not improve upon previous steps.")
            abort_with_error = True

        del callback

    else:
        # just run the test
        update_work_percent(1.0)
        
        callback = ValidationCallback(model, classes, X_test, Y_test, max_epochs, filename, output_prefix, output_path, best_accuracy_worst_class, settings, device)
        if len(X_train) > 0:
            callback.evaluate(0, False)

        del callback
    
    '''except UserCancelException:
        print("Training cancelled by the user.")
    except UserSkipException:
        print("Training skipped by the user.")
    except Exception as e:
        print("Error during training: " + str(e))
        raise e'''

    del X_train
    del Y_train
    del X_test
    del Y_test

    #del train_data
    #del val_data
    del train_loader
    del val_loader

    X = None
    Y = None
    X_val = None
    Y_val = None

    clear_caches()

    if abort_with_error:
        raise Exception("aborting with error")
