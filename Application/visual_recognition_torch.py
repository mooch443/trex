# Standard library imports
import gc
import torch
import json
import os

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchmetrics
from tqdm import tqdm
import datetime

# Local imports
import TRex
from visual_identification_network_torch import ModelFetcher

static_inputs : torch.Tensor = None
static_targets : torch.Tensor = None
loaded_checkpoint : dict = None
loaded_weights : TRex.VIWeights = None

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

def save_pytorch_model_as_jit(model, output_path, metadata, dummy_input=None):
    """
    Converts a PyTorch model to TorchScript and saves it with metadata.

    Parameters:
      model (torch.nn.Module): The PyTorch model to convert.
      output_path (str): Path where the TorchScript model will be saved.
      metadata (dict): Dictionary containing metadata to save with the model.
                       For example:
                       {
                           "input_shape": (width, height, channels),
                           "model_type": "converted from Keras",
                           "num_classes": num_classes,
                           "epoch": None,
                           "uniqueness": None
                       }
      dummy_input (torch.Tensor, optional): A dummy input for model tracing if scripting fails.
    
    Returns:
      torch.jit.ScriptModule: The converted TorchScript model.
    
    Raises:
      ValueError: If scripting fails and no dummy_input is provided for tracing.
    """
    try:
        # Try to convert the model using scripting.
        scripted_model = torch.jit.script(model)
        print("Model scripted successfully.")
    except Exception as e:
        # If scripting fails and a dummy input is provided, fallback to tracing.
        if dummy_input is None:
            raise ValueError("Scripting failed and no dummy input provided for tracing. Error: " + str(e))
        print("Scripting failed, falling back to tracing. Error:", e)
        scripted_model = torch.jit.trace(model, dummy_input)
        print("Model traced successfully.")
    
    # Prepare metadata extra file as JSON.
    extra_files = {"metadata": json.dumps(metadata)}
    
    # Save the scripted (or traced) model along with the extra metadata.
    torch.jit.save(scripted_model, output_path, _extra_files=extra_files)
    print(f"TorchScript model with metadata saved at: {output_path}")
    
    # Optionally, load the model back to verify the metadata was saved.
    files = {"metadata": ""}
    _ = torch.jit.load(output_path, _extra_files=files)
    loaded_metadata = json.loads(files["metadata"])
    print("JIT Loaded metadata:", loaded_metadata)
    
    return scripted_model

def save_model_files(model, output_path, accuracy, suffix='', epoch=None):
    checkpoint = {
        'model': None,
        'state_dict': None,
        'metadata': {
            'input_shape': (image_width, image_height, image_channels),
            'num_classes': len(classes),
            'video_name': TRex.setting("source"),
            'epoch': epoch,
            'uniqueness': accuracy,
        }
    }
    TRex.log(f"# [saving] saving model state dict to {output_path+suffix}.pth")
    try:
        checkpoint['state_dict'] = model.state_dict()
        torch.save(checkpoint, output_path+suffix+".pth")
    except Exception as e:
        TRex.warn("Error saving model: " + str(e))

    TRex.log(f"# [saving] saving complete model to {output_path+suffix}_model.pth")
    try:
        #checkpoint['model'] = model
        #checkpoint['state_dict'] = model.state_dict()
        # save as a jit model
        save_pytorch_model_as_jit(model, output_path+suffix+"_model.pth", checkpoint['metadata'])
        #torch.save(checkpoint, output_path+suffix+"_model.pth")
    except Exception as e:
        TRex.warn("Error saving model: " + str(e))
        
    TRex.log("# [saving] saved states to "+output_path+suffix+" with accuracy of "+str(accuracy))

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
    device = TRex.choose_device()
    TRex.log(f"Clearing caches for {device}...")
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
    else:
        TRex.log(f"No cache to clear {device}")

    gc.collect()

p_softmax = None

def check_device_equivalence(device, model):
    model_device = str(next(model.parameters()).device)
    if model_device != device:
        # check if one of them has a : specifier and the other not
        # if yes, attach :0 to the other
        if ":" in model_device and ":" not in device:
            device += ":0"
        elif ":" in device and ":" not in model_device:
            model_device += ":0"
        if model_device != device:
            raise RuntimeError(f"Model device {model_device} and input device {device} are not the same")

def predict_numpy(model, images, batch_size, device):
    global p_softmax

    assert device is not None, "No device provided"
    assert model is not None, "No model provided"
    assert images is not None, "No images provided"
    assert len(images) > 0, "No images provided"
    assert batch_size > 0, "Invalid batch size"
    assert len(images.shape) == 4, "Invalid image shape"
    assert images.shape[1] == image_height, f"Invalid image height: {images.shape[1]} vs. {image_height}"
    assert images.shape[2] == image_width, f"Invalid image width: {images.shape[2]} vs. {image_width}"
    assert images.shape[3] == image_channels, f"Invalid image channels: {images.shape[3]} vs. {image_channels}"

    # check if the model is also on the same device
    check_device_equivalence(device, model)

    model.eval()

    # predict in batches
    output = []
    with torch.no_grad():
        has_softmax = False
        try:
            #print(f"Model: {model}")
            #print(f"Model children: {list(model.named_children())[-1]}")
            has_softmax = list(model.named_children())[-1][-1].original_name == "Softmax"
        except:
            pass
        #print(f"Model has softmax: {has_softmax}")

        if p_softmax is None:
            p_softmax = nn.Softmax(dim=1).to(device)
        for i in range(0, len(images), batch_size):
            x = torch.tensor(images[i:i+batch_size], dtype=torch.float32, requires_grad=False, device=device).detach()

            # check if the model ends on a softmax layer
            if has_softmax:
                output.append(model(x).cpu().numpy())
                #print("Using model's softmax")
            else:
                #TRex.log(f"Using custom softmax for {x.shape} and {x.dtype} {x.device} {device}")
                output.append(p_softmax(model(x)).cpu().numpy())
                #print("Using custom softmax")
            
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
        
        if update_work_percent is not None:
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
                save_model_files(model=model, output_path=output_path, accuracy=unique, suffix=f"_progress", epoch=epoch)
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

def get_default_network():
    global image_channels
    global image_width, image_height, classes, learning_rate, network_version

    # if no device is specified, use cuda if available, otherwise use mps/cpu
    device = TRex.choose_device()
    assert device is not None, "No device specified"

    loaded_model = ModelFetcher().get_model(network_version, len(classes), image_channels, image_width, image_height, device=device)
    TRex.log(f"Reinitialized network with an empty network version {network_version}.")
    TRex.log(f"Trainable parameters: {sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)}")
    TRex.log(f"Device: {device}")

    return loaded_model

def reinitialize_network():
    global model, loaded_checkpoint, loaded_weights
    model = get_default_network()
    loaded_checkpoint = None
    loaded_weights = None

def get_loaded_weights():
    global loaded_weights
    return loaded_weights.to_string()

# It is assumed that the following globals are defined elsewhere:
#   image_width, image_height, image_channels, classes, model
# and that TRex (with log() and warn() methods) and get_default_network() are available.
# Also, ConfigurationError is defined as follows:
class ConfigurationError(Exception):
    """Raised when the model’s configuration (input dimensions or number of classes)
    does not match the current settings."""
    pass

# New utility function to check checkpoint metadata compatibility.
def check_checkpoint_compatibility(
    metadata: dict,
    context: str = ""
):
    global image_width, image_height, image_channels, classes
    expected_input_shape = (image_width, image_height, image_channels)
    expected_num_classes = len(classes)

    errors = []
    if expected_input_shape is not None and metadata is not None and "input_shape" in metadata:
        # Compare as lists to avoid issues with tuples vs lists.
        if list(metadata["input_shape"]) != list(expected_input_shape):
            if context:
                errors.append(
                    f"Mismatch in input dimensions: {context} expects {expected_input_shape} but checkpoint has {metadata['input_shape']}."
                )
            else:
                errors.append(
                    f"Mismatch in input dimensions: expected {expected_input_shape} but checkpoint metadata has {metadata['input_shape']}."
                )
    if expected_num_classes is not None and metadata is not None and "num_classes" in metadata:
        if metadata["num_classes"] != expected_num_classes:
            if context:
                errors.append(
                    f"Mismatch in number of classes: {context} expects {expected_num_classes} but checkpoint has {metadata['num_classes']}."
                )
            else:
                errors.append(
                    f"Mismatch in number of classes: expected {expected_num_classes} but checkpoint metadata has {metadata['num_classes']}."
                )
    if errors:
        raise ConfigurationError(" ".join(errors))


def load_checkpoint_from_file(file_path: str, device: str):
    """
    Loads a checkpoint from the specified file path.

    The file is expected to be a .pth file that contains a dictionary with:
      - A "model" field (for a complete model) and/or
      - A "state_dict" field (with optional "metadata").

    If metadata is present, it is verified against the current globals:
      (image_width, image_height, image_channels, and len(classes)).

    Returns a checkpoint dict (or wraps a plain state dict).
    """
    if not os.path.exists(file_path):
        raise Exception("Checkpoint file not found at " + file_path)
    #TRex.log(f"Loading checkpoint as JIT from {file_path}...")

    try:
        files = {"metadata": ""}
        cp = torch.jit.load(file_path, map_location=device, _extra_files=files)

        metadata = None
        try:
            metadata = json.loads(files["metadata"])
        except Exception as e:
            TRex.warn("\t- Failed to load metadata from JIT checkpoint: " + str(e))

        cp = {
            "model": cp,
            "metadata": metadata
        }

        TRex.log(f"\t+ Loaded checkpoint from JIT {file_path}.")

    except Exception as e:
        TRex.log(f"\t- Failed to load checkpoint as JIT, trying torch.load.")

        # Fallback to torch.load if JIT load fails.
        from visual_identification_network_torch import (
            PermuteAxesWrapper, Normalize, V118_3, V110, V119, V200
        )
        # Register safe globals for torch.serialization.
        torch.serialization.add_safe_globals([PermuteAxesWrapper, Normalize, transforms.transforms.Normalize])
        torch.serialization.add_safe_globals([set])
        torch.serialization.add_safe_globals([V118_3, V110, V119, V200])
        torch.serialization.add_safe_globals([nn.Softmax, nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                                                nn.MaxPool2d, nn.Linear, nn.Dropout, nn.Dropout2d,
                                                nn.LayerNorm, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                                                nn.AvgPool2d, nn.MaxPool2d, nn.Flatten, nn.Sequential])

        cp = torch.load(file_path, map_location=device, weights_only=True)
        TRex.log(f"\t+ Loaded torch.load checkpoint from {file_path}: {cp.keys()}")

    # If the checkpoint is a dict and contains metadata, perform compatibility checks.
    if isinstance(cp, dict):
        #if "metadata" in cp:
        #    metadata = cp["metadata"]
        #    check_checkpoint_compatibility(metadata)
        return cp
    else:
        TRex.log("\t+ Loaded checkpoint is a plain state dict without metadata.")
        return {"state_dict": cp}


def apply_checkpoint_to_model(target_model: torch.nn.Module, checkpoint):
    """
    Applies weights from the checkpoint to the given target_model,
    checking compatibility based on metadata if available.

    The checkpoint can be:
      - A dict with a "model" field (a complete model) that is not None,
      - A dict with a "state_dict" field (with optional "metadata"),
      - Or a plain state dict.

    Compatibility is verified by comparing checkpoint metadata against attributes
    of the target_model (if they exist, e.g. target_model.input_shape and target_model.num_classes).

    Raises:
        ConfigurationError: If the checkpoint metadata is incompatible with the target model.
        Exception: If loading the weights fails.
    """
    state_dict = None

    if isinstance(checkpoint, dict):
        if "metadata" in checkpoint:
            metadata = checkpoint["metadata"]
            check_checkpoint_compatibility(
                metadata,
                context="target model"
            )
        # Prefer a complete model if available.
        if "model" in checkpoint and checkpoint["model"] is not None:
            TRex.log("The checkpoint has a complete model...")
            try:
                state_dict = checkpoint["model"].state_dict()
            except Exception as e:
                raise Exception("Failed to extract state dict from complete model in checkpoint: " + str(e))
        elif "state_dict" in checkpoint:
            TRex.log("The checkpoint has a state_dict...")
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            TRex.warn("Invalid checkpoint format: missing both 'model' and 'state_dict' keys. Assuming this is only a state_dict.")
    else:
        state_dict = checkpoint

    try:
        if target_model is None:
            target_model = get_default_network()
        target_model.load_state_dict(state_dict)
        TRex.log("Checkpoint weights applied successfully to target model.")
        return target_model
    except Exception as e:
        if "model" not in checkpoint:
            raise e

        TRex.warn("Failed to apply checkpoint weights to target model. Trying to load the model directly: " + str(e))
        target_model = checkpoint["model"]
        TRex.log("Loaded complete model from checkpoint: " + str(target_model))
        return target_model

def load_model_from_file(file_path: str, device: str, new_model: torch.nn.Module = None) -> tuple[torch.nn.Module, dict]:
    """
    Loads a model from the specified checkpoint file and returns a fully initialized PyTorch model.
    
    The function first loads the checkpoint using load_checkpoint_from_file(). If the checkpoint
    contains a complete model in the "model" field (and it passes metadata checks via check_checkpoint_compatibility()),
    that model is returned.
    Otherwise, it instantiates a new model using get_default_network(), applies the checkpoint weights
    to it via apply_checkpoint_to_model(), and returns the updated model.
    """
    cp = load_checkpoint_from_file(file_path, device=device)
    
    # If a complete model is available, try using it.
    if isinstance(cp, dict) and ("model" in cp and cp["model"] is not None):
        try:
            if "metadata" in cp:
                check_checkpoint_compatibility(cp["metadata"])
            TRex.log("Loaded complete model from checkpoint.")
            return cp["model"], cp
        except ConfigurationError as e:
            TRex.warn("Complete model from checkpoint failed compatibility checks: " + str(e) + ". Falling back to state_dict loading")
        except Exception as e:
            TRex.warn("Failed to load complete model from checkpoint: " + str(e))
    
    # Otherwise, load the state_dict into a new model.
    try:
        if new_model is None:
            TRex.log("Instantiating new model...")
            new_model = get_default_network()
        TRex.log(f"Applying checkpoint weights to new model...")
        new_model = apply_checkpoint_to_model(new_model, cp)
        TRex.log("Loaded model from checkpoint state dict.")
        return new_model, cp
    except ConfigurationError as e:
        raise ConfigurationError("Loaded model from checkpoint failed compatibility checks: " + str(e))
    except Exception as e:
        raise Exception("Failed to load model from checkpoint state dict: " + str(e))

def unload_weights():
    global model, loaded_checkpoint, loaded_weights

    TRex.log("Unloading model weights...")
    model = None
    loaded_checkpoint = None
    loaded_weights = None

    clear_caches()

def load_weights(path: str = None) -> str:
    """
    Loads model weights from the specified checkpoint file and applies them to the current global model.
    
    The file is expected to contain either a complete model (in a "model" field) or a "state_dict".
    This function loads the checkpoint and then applies it using apply_checkpoint_to_current_model().
    """
    global output_path, model, loaded_checkpoint, loaded_weights

    if path is None:
        path = output_path

    model = None
    loaded_checkpoint = None
    loaded_weights = None
    modified = None
    device = TRex.choose_device()

    if path.endswith(".pth"):
        saved_path = path
        cp = load_checkpoint_from_file(saved_path, device=device)

    else:
        saved_path = path + "_model.pth"
        try:
            cp = load_checkpoint_from_file(saved_path, device=device)
        except ConfigurationError as e:
            raise ConfigurationError(f"Loaded model from {path}_model.pth failed compatibility checks: {str(e)}")
        except Exception as e:
            saved_path = path+".pth"
            TRex.log(f"Failed to load model from {path}_model.pth ({e}). Trying {saved_path}")
            cp = load_checkpoint_from_file(saved_path, device=device)

    metadata = cp["metadata"] if "metadata" in cp else None

    TRex.log("Loaded checkpoint with metadata: " + str(cp["metadata"] if "metadata" in cp else None))
    model = apply_checkpoint_to_model(model, cp)
    #print("Loaded model weights from checkpoint: ", model)

    if "metadata" in cp and cp["metadata"] is not None and "modified" in cp["metadata"]:
        modified = cp["metadata"]["modified"]
    else:
        try:
            # get modified time as a unix timestamp
            modified = int(os.path.getmtime(saved_path))
        except Exception as e:
            TRex.warn(f"Failed to get modified time for {saved_path}: {str(e)}")

    loaded_checkpoint = cp
    loaded_weights = TRex.VIWeights(
        path = saved_path,
        uniqueness = metadata["uniqueness"] if metadata is not None and "uniqueness" in metadata else None,
        status = "FINISHED",
        modified = modified,
        loaded = True,
        resolution = TRex.DetectResolution(metadata["input_shape"][:2]) if metadata is not None and "input_shape" in metadata else None,
        classes = metadata["num_classes"] if metadata is not None and "num_classes" in metadata else None
    )

    return loaded_weights.to_string()


def find_available_weights(path: str = None) -> str:
    """
    Searches for available model weights in the specified directory and returns a JSON array of serialized TRex.VIWeights strings.
    
    Only files matching the allowed pattern are processed. In this implementation, only files with names exactly matching
    either <path>_model.pth or <path>.pth are allowed.
    
    Returns:
        A JSON-encoded array (string) of serialized TRex.VIWeights objects. If no valid weight files are found, an empty JSON array ("[]") is returned.
    """
    global output_path
    device = TRex.choose_device()

    if path is None:
        path = output_path

    # check the extension of this file and see whether it is a .pth file
    if path.endswith(".pth"):
        candidate_files = [path]
    else:
        candidate_files = [
            path + "_model.pth",
            path + ".pth"
        ] #glob.glob(pattern)
    
    serialized_weights = []
    
    for file_path in candidate_files:
        # Only process files that match the allowed regex.
        #if not allowed_pattern.match(os.path.basename(file_path)):
        #    continue
        TRex.log(f"Checking candidate file {file_path}...")
        
        try:
            cp = load_checkpoint_from_file(file_path, device=device)
            #TRex.log(f"\t+ Loaded checkpoint.")
        except Exception as e:
            TRex.warn(f"\t- Failed to load model from {file_path}: {str(e)}")
            continue

        metadata = cp.get("metadata", None)
        modified = None
        try:
            # get modified time as a unix timestamp
            modified = int(os.path.getmtime(file_path))
            TRex.log(f"\t+ Got modified time from file: {modified}")
        except Exception as e:
            TRex.warn(f"\t- Failed to get modified time for {file_path}: {str(e)}")

        serialized_weights.append(TRex.VIWeights(
            path=file_path,
            uniqueness=metadata["uniqueness"] if metadata is not None and "uniqueness" in metadata else None,
            status="FINISHED",
            modified=modified,
            loaded=loaded_weights.path.str() == file_path if loaded_weights is not None else False,
            resolution=TRex.DetectResolution(metadata["input_shape"][:2]) if metadata is not None and "input_shape" in metadata else None,
            classes=metadata["num_classes"] if metadata is not None and "num_classes" in metadata else None
        ).to_json())

    TRex.log(f"Found {len(serialized_weights)} available weights: {serialized_weights}")
    return json.dumps(serialized_weights)

def predict():
    global receive, images, model, image_channels, batch_size, image_width, image_height

    device = TRex.choose_device()
    assert device is not None, "No device available for prediction."
    assert model is not None, "No model available for prediction."
    assert images is not None, "No images available for prediction."

    images = np.array(images, copy=False)
    assert len(images.shape) == 4, f"error with the shape {images.shape} < len 4"
    assert images.shape[3] == image_channels, f"error with the shape {images.shape} < channels {image_channels}"
    assert images.shape[1] == image_width and images.shape[2] == image_height, f"error with the shape {images.shape} < width {image_width} height {image_height}"

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

    output = predict_numpy(model, images, batch_size, device=device).astype(np.float32)#.cpu().numpy()
    TRex.log(f"Predicted images with shape {output.shape}")
    
    receive(output, indexes)

    del output
    del indexes
    del images

    gc.collect()

def train(model, train_loader, val_loader, criterion, optimizer : torch.optim.Adam, callback : ValidationCallback, scheduler, transform, settings, device):
    global get_abort_training
    global static_inputs, static_targets
    num_classes = len(settings["classes"])

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
                static_inputs = inputs.clone()
                static_targets = targets.clone()
                #static_inputs.resize_(inputs.shape).copy_(inputs)
                #static_targets.resize_(targets.shape).copy_(targets)

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
    

def start_learning():
    global image_channels, output_prefix, filename
    global best_accuracy_worst_class, max_epochs, image_width, image_height, update_work_percent
    global output_path, classes, learning_rate, accumulation_step, global_tracklet, verbosity
    global batch_size, X_val, Y_val, X, Y, run_training, save_weights_after, do_save_training_images, min_iterations
    global get_abort_training, model, train, network_version, loaded_checkpoint, loaded_weights

    device = TRex.choose_device()
    assert device is not None

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
    val_loader = CustomDataLoader(X_test, Y_test, batch_size=batch_size, device='cpu')

    settings["model"] = network_version
    settings["device"] = str(device)

    #if model is None:
    #    TRex.log(f"# [init] loading model {model_name} with {num_classes} classes and {image_channels} channels ({image_width}x{image_height})")
    #    model = model_fetcher.get_model(model_name, num_classes, image_channels, image_width, image_height, device=device)

    assert model is not None, "Model is not initialized."

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
                model, cp = load_model_from_file(output_path+'_progress.pth', device=device, new_model = model)
                loaded_checkpoint = cp
                loaded_weights = TRex.VIWeights(
                    path = output_path+'_progress.pth',
                    uniqueness = cp["metadata"]["uniqueness"] if "uniqueness" in cp["metadata"] else None,
                    status = "FINISHED",
                    modified = cp["metadata"]["modified"] if "modified" in cp["metadata"] else None,
                    loaded = True,
                    resolution = TRex.DetectResolution(cp["metadata"]["input_shape"][:2]) if "input_shape" in cp["metadata"] else None,
                    classes = cp["metadata"]["num_classes"] if "num_classes" in cp["metadata"] else None
                )
                #apply_checkpoint_to_model(model, checkpoint)
            except Exception as e:
                TRex.warn(str(e))
                raise Exception("Failed to load weights. This is likely because either the individual_image_size is different, or because existing weights might have been generated by a different version of TRex (see visual_identification_version).")
            
            np.savez(history_path, #history=np.array(history.history, dtype="object"), 
                     uniquenesses=callback.uniquenesses, better_values=callback.better_values, much_better_values=callback.much_better_values, worst_values=callback.worst_values, mean_values=callback.mean_values, settings=np.array(settings, dtype="object"), per_class_accuracy=np.array(callback.per_class_accuracy),
                samples_per_class=np.array(per_class, dtype="object"))

            if save_weights_after:
                save_model_files(model=model, output_path=output_path, accuracy=callback.best_result["unique"])
        
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
        if update_work_percent is not None:
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
    else:
        return loaded_weights.to_string()
