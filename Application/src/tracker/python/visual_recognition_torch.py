"""Visual recognition training/inference utilities for TRex.

This module provides:
- Dataset/wrapper utilities that operate on NHWC images in [0,255]
- Thread-backed prefetching for DataLoader iteration
- Training loop with AMP, validation callback, and early stopping via a
  project-specific "uniqueness" metric
- Checkpoint save/load helpers supporting both state_dict and TorchScript

External callers generally pass images as NumPy arrays (HxWxC), while models
internally consume NCHW tensors via a wrapper defined in
`visual_identification_network_torch`.
"""

# Standard library imports
import gc
import torch
import json
import os
import threading
from queue import Queue

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from tqdm import tqdm
from typing import Optional

from trex_utils import UserCancelException, UserSkipException, save_pytorch_model_as_jit

# Local imports
import os
try:
    import TRex  # embedded module from the host app
except Exception:
    # Worker processes (spawn) can’t see the embedded module. Use a stub.
    class _TRexStub:
        @staticmethod
        def log(*args, **kwargs): pass  # or print("[TRex]", *args)
        @staticmethod
        def warn(*args, **kwargs): pass
        @staticmethod
        def choose_device(): return os.environ.get("TREX_DEFAULT_DEVICE", "cpu")
        @staticmethod
        def setting(_name, _default=None): return _default
    TRex = _TRexStub()

from visual_identification_network_torch import ModelFetcher
import trex_utils
from trex_utils import _first_shape, _as_batched_np, load_checkpoint_from_file, check_checkpoint_compatibility, ConfigurationError

static_inputs : torch.Tensor = None
static_targets : torch.Tensor = None
loaded_checkpoint : dict = None
loaded_weights : TRex.VIWeights = None
p_softmax = None
output_path : Optional[str] = None
X : Optional[list[np.ndarray]] = None
Y : Optional[list[int]] = None
X_val : Optional[list[np.ndarray]] = None
Y_val : Optional[list[int]] = None

def _dbg_enabled() -> bool:
    """Return True if verbose debug logging is enabled via TREX_DEBUG_TRAIN=1."""
    return os.environ.get("TREX_DEBUG_TRAIN", "0") == "1"

def _dbg_tensor(name: str, t: torch.Tensor | None):
    """Log basic properties of a tensor when debug is enabled."""
    if not _dbg_enabled() or t is None:
        return
    try:
        TRex.log(f"[DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} contiguous={t.is_contiguous()} stride={t.stride()} req_grad={t.requires_grad}")
    except Exception as e:
        TRex.warn(f"[DBG] failed to inspect {name}: {e}")

def _dbg_model(name: str, model: nn.Module | None):
    """Log model name, parameter counts, and device when debug is enabled."""
    if not _dbg_enabled() or model is None:
        return
    try:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_device = str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "cpu"
        TRex.log(f"[DBG] {name}: {model.__class__.__name__} with {num_params} params ({num_trainable} trainable) on {model_device}")
    except Exception as e:
        TRex.warn(f"[DBG] failed to inspect {name}: {e}")

def save_model_files(model, output_path, accuracy, suffix='', epoch=None):
    """Persist the model in two forms and attach training metadata.

    - Writes `<output_path><suffix>_dict.pth`: Python checkpoint with `state_dict` and `metadata`.
    - Writes `<output_path><suffix>_model.pth`: TorchScript export via `save_pytorch_model_as_jit`.

    Metadata includes input_shape (W,H,C), num_classes, video_name, epoch,
    uniqueness (float), and model_type.
    """
    checkpoint = {
        'model': None,
        'state_dict': None,
        'metadata': {
            'input_shape': (image_width, image_height, image_channels),
            'num_classes': len(classes),
            'video_name': TRex.setting("source"),
            'epoch': epoch,
            'uniqueness': accuracy,
            'model_type': str(network_version),
        }
    }
    TRex.log(f"# [saving] saving model state dict to {output_path+suffix}_dict.pth")
    try:
        checkpoint['state_dict'] = model.state_dict()
        torch.save(checkpoint, output_path+suffix+"_dict.pth")
    except Exception as e:
        TRex.warn("Error saving model: " + str(e))

    TRex.log(f"# [saving] saving complete model to {output_path+suffix}_model.pth")
    try:
        #checkpoint['model'] = model
        #checkpoint['state_dict'] = model.state_dict()
        # save as a jit model
        save_pytorch_model_as_jit(model, output_path+suffix+"_model.pth", checkpoint['metadata'])
    except Exception as e:
        TRex.warn("Error saving complete model: " + str(e))
        
    TRex.log("# [saving] saved states to "+output_path+suffix+" with accuracy of "+str(accuracy))

# alternative for onehotencoder from sklearn:
class OneHotEncoder:
    """Minimal OneHotEncoder replacement for integer labels (no sklearn dependency)."""
    def __init__(self, sparse_output = False):
        self.categories_ = None

    def fit(self, y):
        """Discover sorted unique categories from `y` and return self."""
        self.categories_ = np.unique(y).tolist()
        return self

    def transform(self, y : np.ndarray):
        """Convert labels in `y` to a dense one-hot ndarray using discovered categories."""
        if self.categories_ is None:
            raise RuntimeError("You must fit the encoder before transforming data.")

        num_classes = len(self.categories_)
        one_hot = np.zeros((len(y), num_classes))
        for i, c in enumerate(y):
            one_hot[i, self.categories_.index(c)] = 1
        return one_hot

    def fit_transform(self, y : np.ndarray):
        """Fit on `y` then return its one-hot encoding."""
        return self.fit(y).transform(y)

class TRexImageDataset(Dataset):
    """Torch Dataset that yields NHWC float32 images in [0,255] and int64 labels.

    - Accepts X as np.ndarray (N,H,W,C) or list of HxWxC ndarrays
    - Applies transforms on CHW float in [0,1], then converts back to NHWC [0,255]
    """
    def __init__(self, X, Y, transform: transforms.Compose | None = None, device = None):
        self.X = X
        self.Y = Y if isinstance(Y, np.ndarray) else np.asarray(Y, dtype=np.int64)
        self.transform = transform
        self.device = device

    def __len__(self):
        """Return N derived from `X` (len(X) for sequences; X.shape[0] for ndarrays)."""
        return len(self.X) if not isinstance(self.X, np.ndarray) else self.X.shape[0]

    def __getitem__(self, idx):
        """Return a tuple `(x, y)` where:
        - `x` is an NHWC float32 tensor in [0,255] (transforms applied on CHW [0,1])
        - `y` is an int class index
        """
        if isinstance(self.X, np.ndarray):
            im = self.X[idx]
        else:
            im = self.X[idx]
        # im: HWC uint8/float; convert to CHW float in [0,1] for transforms
        x = torch.from_numpy(im).to(dtype=torch.float32)
        x = x.permute(2, 0, 1).contiguous().clone()  # CHW contiguous
        x = x.div(255.0)
        if self.transform is not None:
            x = self.transform(x)
        # Convert back to NHWC in [0,255] to preserve existing model path
        x = (x.clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).contiguous().clone()  # HWC contiguous
        #TRex.log(f"Dataset idx {idx}: transformed image shape {self.Y.shape} dtype {self.Y.dtype} device {self.Y.device if isinstance(self.Y, torch.Tensor) else 'cpu'}")
        y = int(self.Y[idx])
        # Return a Tensor (not NumPy) to keep strides and contiguity under control
        return x, y

# --- ThreadedLoader: thread-backed prefetcher for any iterable loader ---
class ThreadedLoader:
    """A light-weight, thread-backed prefetcher for any Python iterable loader.

    Purpose:
      - Avoids multiprocessing (safe with embedded/pybind11 modules)
      - Overlaps data preparation with GPU compute via a background thread
      - Preserves the original loader's batching/collation

    Usage:
      wrapped = ThreadedLoader(train_loader, max_prefetch=2)
      for batch in wrapped: ...
    """
    def __init__(self, loader, max_prefetch: int = 2):
        self.loader = loader
        self.max_prefetch = max(1, int(max_prefetch))
        self._queue: Queue | None = None
        self._thread: threading.Thread | None = None
        self._sentinel = object()
        self._exc: BaseException | None = None

    def __len__(self):
        return len(self.loader)

    def _producer(self, it):
        try:
            for item in it:
                self._queue.put(item)
            # normal end
            self._queue.put(self._sentinel)
        except BaseException as e:
            # store exception and signal end; will be re-raised on consumer side
            self._exc = e
            self._queue.put(self._sentinel)

    def __iter__(self):
        it = iter(self.loader)
        self._queue = Queue(self.max_prefetch)
        self._exc = None
        self._thread = threading.Thread(target=self._producer, args=(it,), daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            # join the thread quickly and propagate producer error if any
            t = self._thread
            self._thread = None
            if t is not None and t.is_alive():
                t.join(timeout=0.1)
            if self._exc is not None:
                e = self._exc
                self._exc = None
                raise e
            raise StopIteration
        return item

def clear_caches():
    """Free device-specific caches (CUDA/MPS) and run Python GC."""
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

def check_device_equivalence(device, model):
    """Ensure `model` parameters live on the same device string as `device`.

    Normalizes optional `:idx` suffix differences before comparing.
    """
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
    """Run batched inference and return class probabilities as np.ndarray (N,C).

    - Accepts list/array of NHWC images in [0,255]; validates against configured W,H,C.
    - Ensures model/device match; uses model's terminal Softmax if present,
      otherwise applies a cached `nn.Softmax(dim=1)` to logits.
    """
    global p_softmax

    assert device is not None, "No device provided"
    assert model is not None, "No model provided"
    assert images is not None, "No images provided"
    assert batch_size > 0, "Invalid batch size"

    N = len(images)
    assert N > 0, "No images provided"

    H, W, C = _first_shape(images)
    assert H == image_height, f"Invalid image height: {H} vs. {image_height}"
    assert W == image_width,  f"Invalid image width: {W} vs. {image_width}"
    assert C == image_channels, f"Invalid image channels: {C} vs. {image_channels}"

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
            j = min(i + batch_size, N)

            #x = torch.tensor(images[i:i+batch_size], dtype=torch.float32, requires_grad=False, device=device)
            batch_np = _as_batched_np(images, slice(i, j))   # ndarray (B,H,W,C). One host copy iff list
            # Ensure float32 for MPS compatibility
            x = torch.from_numpy(batch_np).to(device, dtype=torch.float32) # Tensor (B,H,W,C)

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
    """Validation and early-stopping orchestrator using per-class accuracy and a 'uniqueness' score.

    - Evaluates X_test/Y_test (one-hot) grouped by class; records worst/mean accuracy history.
    - Saves interim weights when uniqueness improves and exposes a stop flag.
    - Stops when uniqueness passes acceptance thresholds, worst-class accuracy
      saturates, or loss trends indicate overfitting (relative to `patience` and `compare_acc`).
    """
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
            #TRex.log(f"Y_test shape: {Y_test.shape} dtype: {Y_test.dtype} device: {Y_test.device if isinstance(Y_test, torch.Tensor) else 'cpu'}")
            labels = Y_test.argmax(axis=1) if len(Y_test) > 0 else np.zeros((0,), dtype=np.int64)
            for c in classes:
                mask = (labels == c)
                if isinstance(X_test, np.ndarray):
                    a = X_test[mask]
                else:
                    idxs = np.nonzero(mask)[0]
                    a = [X_test[k] for k in idxs]
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

            # shape of y: (N, num_classes)
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
        """Format and push an epoch status line (best/prev. uniqueness and coarse loss change) to the TRex UI."""
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
        """Evaluate on the validation split, update metrics/history, and maybe save.

        Returns the current uniqueness estimate. When `save` is True, updates
        early-stopping state and may persist a progress checkpoint.
        """
        global update_work_percent, set_stop_reason, set_per_class_accuracy, set_uniqueness_history, estimate_uniqueness, acceptable_uniqueness, accepted_uniqueness

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
        """Hook called after each epoch to run evaluation and handle user abort/skip."""
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
        """Hook called after a training batch to update progress and check abort/skip."""
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
    """Instantiate and return the selected model via ModelFetcher on the active device."""
    global image_channels
    global image_width, image_height, classes, learning_rate, network_version

    # if no device is specified, use cuda if available, otherwise use mps/cpu
    device = TRex.choose_device()
    assert device is not None, "No device specified"

    loaded_model = ModelFetcher().get_model(network_version, len(classes), image_channels, image_width, image_height, device=device)
    TRex.log(f"Reinitialized network with an empty network version {network_version} @ {image_channels}.")
    TRex.log(f"Trainable parameters: {sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)}")
    TRex.log(f"Device: {device}")

    return loaded_model

def reinitialize_network():
    """Reset the global `model` to a fresh architecture and clear loaded weights."""
    global model, loaded_checkpoint, loaded_weights
    model = get_default_network()
    loaded_checkpoint = None
    loaded_weights = None

def get_loaded_weights():
    """Return the serialized string representation of the currently loaded weights."""
    global loaded_weights
    return loaded_weights.to_string()


def apply_checkpoint_to_model(target_model: torch.nn.Module, checkpoint):
    """
    Applies weights from the checkpoint to the given target_model,
    checking compatibility based on metadata if available.

    The checkpoint can be:
      - A dict with a "state_dict" field (preferred)
      - A dict with a "model" field (TorchScript or nn.Module); we will extract its state_dict

    Compatibility is verified against the current training configuration
    (image size, channels, classes) using checkpoint metadata when available.
    """
    global image_width, image_height, image_channels, classes

    state_dict = None

    if isinstance(checkpoint, dict):
        if "metadata" in checkpoint:
            metadata = checkpoint["metadata"]
            check_checkpoint_compatibility(
                image_width=image_width,
                image_height=image_height,
                image_channels=image_channels,
                classes=classes,
                metadata=metadata,
                context="target model"
            )
        # Prefer an explicit state_dict over a serialized model
        if "state_dict" in checkpoint and checkpoint["state_dict"] is not None:
            TRex.log("The checkpoint has a state_dict...")
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and checkpoint["model"] is not None:
            TRex.log("Extracting state_dict from checkpoint model...")
            try:
                state_dict = checkpoint["model"].state_dict()
            except Exception as e:
                raise Exception("Failed to extract state dict from model in checkpoint: " + str(e))
        else:
            state_dict = checkpoint
            TRex.warn("Invalid checkpoint format: missing both 'model' and 'state_dict' keys. Assuming this is only a state_dict.")
    else:
        state_dict = checkpoint

    try:
        if target_model is None:
            target_model = get_default_network()
        
        # Load in relaxed mode to tolerate BN↔GN or minor head changes
        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            TRex.warn(f"load_state_dict(strict=False): missing={missing}, unexpected={unexpected}")
        TRex.log("Checkpoint weights applied successfully to target model.")
        return target_model
    except Exception as e:
        raise Exception("Failed to load state dict into target model: " + str(e))

def load_model_from_file(file_path: str, device: str, new_model: torch.nn.Module = None) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint and return a trainable model when possible.

    Preferred path: instantiate the current-code model and apply the checkpoint's
    `state_dict` (after compatibility checks). Fallback: use a serialized model
    from the checkpoint or sibling `*_model.pth` (may be inference-only).

    Returns: (model, checkpoint_dict).
    """
    global image_width, image_height, image_channels, classes

    cp = load_checkpoint_from_file(file_path, device=device)

    # Preferred path: apply state_dict into a fresh model (trainable)
    try:
        if new_model is None:
            TRex.log("Instantiating new model...")
            new_model = get_default_network()
        TRex.log("Applying checkpoint weights to new model (state_dict-first)...")
        new_model = apply_checkpoint_to_model(new_model, cp)
        TRex.log("Loaded model from checkpoint state dict.")
        return new_model, cp
    except ConfigurationError as e:
        TRex.warn("Compatibility check failed for state_dict path: " + str(e))
    except Exception as e:
        TRex.warn("State_dict application failed: " + str(e))

    # Fallback path: use serialized model if available
    ts_cp = None
    if isinstance(cp, dict) and cp.get("model", None) is not None:
        ts_cp = cp
    else:
        # Try sibling "*_model.pth" path if provided file was weights-only
        try:
            if file_path.endswith(".pth") and not file_path.endswith("_model.pth"):
                ts_path = file_path[:-4] + "_model.pth"
                TRex.log(f"Trying serialized-model fallback from {ts_path}...")
                ts_cp = load_checkpoint_from_file(ts_path, device=device)
        except Exception as e2:
            TRex.warn("Serialized-model fallback load failed: " + str(e2))

    if isinstance(ts_cp, dict) and ts_cp.get("model", None) is not None:
        ts_model = ts_cp["model"]
        try:
            # Ensure correct device; jit.load already mapped via device, but `.to` is safe.
            ts_model = ts_model.to(device)
        except Exception:
            pass
        TRex.warn("Using serialized model fallback (may be untrainable).")
        return ts_model, ts_cp

    # No viable fallback
    raise Exception("Failed to load model: state_dict path failed and no serialized-model fallback available.")

def unload_weights():
    """Clear the current model/weights and free caches."""
    global model, loaded_checkpoint, loaded_weights

    TRex.log("Unloading model weights...")
    model = None
    loaded_checkpoint = None
    loaded_weights = None

    clear_caches()

def load_weights(path: Optional[str] = None) -> str:
    """
    Load weights with a simple ordered fallback:
      1) <base>_dict.pth (preferred weights-only)
      2) <base>.pth (legacy weights-only)
      3) <base>_model.pth (serialized full model)
    Tries to apply state_dict to a fresh Python model; otherwise adopts the serialized model.
    """
    global output_path, model, loaded_checkpoint, loaded_weights

    if path is None:
        path = output_path

    model = None
    loaded_checkpoint = None
    loaded_weights = None
    device = TRex.choose_device()

    # Build candidate list
    candidates: list[str]
    if path.endswith(".pth"):
        candidates = [path]
    else:
        candidates = [path + "_dict.pth", path + ".pth", path + "_model.pth"]

    chosen_cp = None
    chosen_path = None
    last_error = None

    for cand in candidates:
        try:
            cp = load_checkpoint_from_file(cand, device=device)
        except Exception as e:
            TRex.log(f"Failed to load from {cand}: {e}")
            last_error = e
            continue

        # Try state_dict application first
        try:
            model_candidate = apply_checkpoint_to_model(model, cp)
            model = model_candidate
            chosen_cp = cp
            chosen_path = cand
            break
        except Exception as e_apply:
            TRex.warn(f"State_dict application failed for {cand}: {e_apply}")
            # If cp has a serialized model, adopt it as inference-only fallback
            if isinstance(cp, dict) and cp.get("model", None) is not None:
                try:
                    model = cp["model"].to(device)
                except Exception:
                    model = cp["model"]
                chosen_cp = cp
                chosen_path = cand
                TRex.warn("Using serialized model (inference-only).")
                break
            # else continue to next candidate

    if chosen_cp is None or chosen_path is None or model is None:
        raise Exception(f"No usable checkpoint found. Last error: {last_error}")

    metadata = chosen_cp.get("metadata", None) if isinstance(chosen_cp, dict) else None

    # Determine modified time
    try:
        modified = metadata.get("modified") if (metadata and "modified" in metadata) else int(os.path.getmtime(chosen_path))
    except Exception:
        modified = None

    loaded_checkpoint = chosen_cp
    loaded_weights = TRex.VIWeights(
        path=chosen_path,
        uniqueness=metadata["uniqueness"] if metadata is not None and "uniqueness" in metadata else None,
        status="FINISHED",
        modified=modified,
        loaded=True,
        resolution=TRex.DetectResolution(metadata["input_shape"][:2]) if metadata is not None and "input_shape" in metadata else None,
        classes=metadata["num_classes"] if metadata is not None and "num_classes" in metadata else None
    )

    return loaded_weights.to_string()


def find_available_weights(path: str = None) -> str:
    """List sibling weight files for a base path and return a JSON array of serialized TRex.VIWeights strings.

    Recognized (priority order): `<path>_dict.pth`, `<path>_model.pth`, `<path>.pth`.
    Each array element is itself a JSON string; the `loaded` field is true for the currently loaded path.
    Returns "[]" if nothing valid is found.
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
            path + "_dict.pth",
            path + "_model.pth",
            path + ".pth"
        ]
    
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
    """Predict class probabilities for global `images` and pass to `receive`.

    Validates NHWC shape against configured (W,H,C), runs `predict_numpy`, and
    emits indexes [0..N-1]. Side effect: calls `receive(output, indexes)`.
    """
    global receive, images, model, image_channels, batch_size, image_width, image_height

    device = TRex.choose_device()
    assert device is not None, "No device available for prediction."
    assert model is not None, "No model available for prediction."
    assert images is not None, "No images available for prediction."

    # Normalize to a Python list of per-image ndarrays
    #images = images if isinstance(images, list) else list(images)
    assert isinstance(images, list), "Images must be provided as a list of NumPy arrays."

    N = len(images)
    assert N > 0, "No images available for prediction."

    # Validate a representative sample (first + up to 7 more) for shape consistency
    first = images[0]
    assert isinstance(first, np.ndarray), "Each image must be a NumPy array"
    assert first.ndim == 3, f"Each image must be HxWxC, got ndim={first.ndim}"
    H, W, C = first.shape
    assert C == image_channels, f"Channel mismatch: {C} vs expected {image_channels}"
    assert (W, H) == (image_width, image_height), (
        f"Size mismatch: {(W, H)} vs expected {(image_width, image_height)}"
    )

    for k in range(1, min(8, N)):
        imk = images[k]
        if not isinstance(imk, np.ndarray) or imk.shape != (H, W, C):
            raise AssertionError(
                f"Inconsistent image at index {k}: got {None if not isinstance(imk, np.ndarray) else imk.shape}, expected {(H, W, C)}"
            )

    TRex.log(f"Predicting {N} images with shape ({N}, {H}, {W}, {C})")

    # Build indexes and run prediction; predict_numpy handles list-of-arrays by stacking per batch
    indexes = np.arange(N, dtype=np.float32)
    output = predict_numpy(model, images, batch_size, device=device).astype(np.float32, copy=False)
    TRex.log(f"Predicted images with shape {output.shape}")

    receive(output, indexes)

    # Cleanup
    del output
    del indexes
    del images
    gc.collect()

def train(model, train_loader, val_loader, criterion, optimizer : torch.optim.Adam, callback : ValidationCallback, scheduler, settings, device):
    """Mini-batch training loop with AMP and threaded prefetch.

    Consumes NHWC float inputs in [0,255] with integer labels; trains with
    CrossEntropy over logits; periodically evaluates on `val_loader`; reports
    progress via `callback` and respects its early-stopping signal.
    """
    global get_abort_training
    num_classes = len(settings["classes"])

    # Initialize metric placeholders (precision/recall currently disabled)
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)

    best_val_acc = 0.0
    # No softmax for training metrics; argmax on logits is identical.

    static_inputs = None
    static_targets = None
    
    # Enable kernel autotuning / TF32 where applicable
    try:
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # AMP (CUDA only)
    from contextlib import nullcontext
    use_amp = (device == 'cuda' and torch.cuda.is_available())
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    try:
        # Prefer new torch.amp API where available
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Optional anomaly detection for better error traces
    if os.environ.get("TREX_TORCH_ANOMALY", "0") == "1":
        try:
            torch.autograd.set_detect_anomaly(True)
            TRex.log("[DBG] Enabled torch.autograd.set_detect_anomaly(True)")
        except Exception as e:
            TRex.warn(f"[DBG] Failed to enable anomaly detection: {e}")
    
    #import tracemalloc;
    #tracemalloc.start(50)

    for epoch in range(settings["epochs"]):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        #snapshot0 = tracemalloc.take_snapshot()
        #import torch.autograd.profiler as profiler

        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        # Wrap DataLoader with ThreadedLoader to prefetch in background
        it_loader = ThreadedLoader(train_loader, max_prefetch=int(os.environ.get("TREX_PREFETCH", "2")))
        
        for batch, (inputs, targets) in tqdm(enumerate(it_loader), total=len(it_loader)):
            #inputs = transform(inputs.permute(0, 3, 1, 2) / 255).permute(0, 2, 3, 1) * 255
            assert isinstance(inputs, torch.Tensor), f"Expected inputs to be a torch.Tensor, got {type(inputs)}"
            assert isinstance(targets, torch.Tensor), f"Expected targets to be a torch.Tensor, got {type(targets)}"
            assert inputs.ndim == 4, f"Expected inputs to be 4D (N,H,W,C), got {inputs.ndim}D"
            assert targets.ndim == 1, f"Expected targets to be 1D (N,), got {targets.ndim}D"
            assert inputs.shape[0] == targets.shape[0], f"Batch size mismatch: {inputs.shape[0]} vs {targets.shape[0]}"
            assert inputs.shape[1] == settings["image_height"] and inputs.shape[2] == settings["image_width"], f"Input size mismatch: {(inputs.shape[1], inputs.shape[2])} vs {(settings['image_height'], settings['image_width'])}"
            assert inputs.shape[3] == settings["image_channels"], f"Input channels mismatch: {inputs.shape[3]} vs {settings['image_channels']}"
            assert targets.dtype in (torch.int64, torch.int32), f"Expected targets to be integer class indices, got {targets.dtype}"
            assert targets.min() >= 0 and targets.max() < num_classes, f"Target class indices out of range: min {targets.min().item()} max {targets.max().item()} vs num_classes {num_classes}"
            
            static_inputs = inputs.to(device, non_blocking=True).contiguous()
            static_targets = targets.to(device, non_blocking=True).long().contiguous()

            if batch == 0 and epoch == 0:
                print(f"Batch {batch}/{len(train_loader)} - inputs: {inputs.shape} targets: {targets.shape}")

                import torchvision
                grid = torchvision.utils.make_grid(inputs.permute(0, 3, 1, 2), nrow=16, value_range=(0,255))
                image = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                TRex.log(f"Image shape: {image.shape}")
                TRex.imshow("batch0", image)
                #plt.imshow(grid.permute(1, 2, 0).numpy().astype(int))
                #plt.show()

            if _dbg_enabled():
                _dbg_tensor("batch.inputs", static_inputs)
                _dbg_tensor("batch.targets", static_targets)

                # print model information
                _dbg_model("batch.model", model)
            
            assert static_inputs.is_contiguous()
            assert static_targets.is_contiguous()

            try:
                with amp_ctx():
                    outputs = model(static_inputs)
                    if _dbg_enabled():
                        _dbg_tensor("batch.outputs", outputs)
                    
                    loss = criterion(outputs.contiguous(), static_targets)
            except Exception as e:
                # Print diagnostics and re-raise
                _dbg_tensor("EX.inputs", static_inputs)
                TRex.warn(f"[DBG] Exception during forward/loss: {e}")
                raise

            try:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            except Exception as e:
                _dbg_tensor("EX.outputs", outputs)
                TRex.warn(f"[DBG] Exception during backward/step: {e}")
                raise

            with torch.no_grad():
                # Compute accuracy from logits directly (no softmax needed)
                pred = outputs.argmax(dim=1)
                acc = torch.mean((pred == static_targets).float())
            
            running_acc += acc
            running_loss += loss.item()

            logs = {'loss': loss.item(), 'acc': acc}
            callback.on_batch_end(batch, logs)

            del inputs
            del targets
            del loss
            del acc
            del logs

            #gc.collect()

        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        #gc.collect()

        running_loss /= len(train_loader)
        acc = running_acc / len(train_loader)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        val_loss = 0
        correct = 0
        total = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for batch, (inputs, targets) in enumerate(val_loader):
                    static_inputs = inputs.to(device).contiguous()
                    static_targets = targets.to(device).long().contiguous()

                    # Validation forward pass
                    with amp_ctx():
                        outputs = model(static_inputs)

                    loss = criterion(outputs.contiguous(), static_targets)
                    val_loss += loss.item()

                    # Calculate the accuracy using logits
                    predicted = outputs.detach().argmax(1)

                    total += static_targets.size(0)
                    correct += predicted.eq(static_targets).sum().item()

                    # Update precision and recall
                    #precision.update(predicted, static_targets.argmax(dim=1))
                    #recall.update(predicted, static_targets.argmax(dim=1))

                    del outputs
                    del loss
                    del predicted

            del static_inputs
            del static_targets

            # Compute average loss and accuracy
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            # Precision/recall placeholders (computation disabled)
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
    
    del precision
    del recall

    #tracemalloc.stop()

    clear_caches()
    

def start_learning():
    """Prepare datasets/loaders, configure augmentation/scheduler, and run training or evaluation.

    Uses globals (X/Y/X_val/Y_val, classes, image size, etc.), optionally saves
    training images, persists progress/history, and returns the serialized
    `loaded_weights` string. Raises if training is required but fails to improve.
    """
    global image_channels, output_prefix, filename
    global best_accuracy_worst_class, max_epochs, image_width, image_height, update_work_percent
    global output_path, classes, learning_rate, accumulation_step, global_tracklet, verbosity
    global batch_size, X_val, Y_val, X, Y, run_training, do_save_training_images, min_iterations
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
        #"min_acceptable_value": 0.98
    }

    # Data augmentation setup
    transform = transforms.Compose([
        #transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=5, translate=(move_range, move_range)),
        transforms.ColorJitter(brightness=(0.85, 1.15),
                                contrast=(0.85, 1.15),
                                saturation=(0.85, 1.15),
                                hue=(-0.05, 0.05),
                               ),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomResizedCrop((image_height, image_width), scale=(0.95, 1.05)),
    ])

    # Expect a list of per-image ndarrays (HxWxC)
    assert isinstance(X, list)
    X_train = X  # keep as sequence; batching happens per step
    Y_train = trex_utils.asarray(Y, dtype=int)

    assert isinstance(X_val, list)
    X_test = X_val
    Y_test = trex_utils.asarray(Y_val, dtype=int)

    # Keep integer class indices for training; build one-hot only for validation callback
    TRex.log(f"Y_test: {Y_test.shape}")
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Y_test_one_hot = onehot_encoder.fit_transform(Y_test.reshape(-1, 1).astype(np.float32))

    def _len_images(arr):
        return len(arr) if not isinstance(arr, np.ndarray) else arr.shape[0]

    def _shape_str(arr):
        return str(arr.shape) if isinstance(arr, np.ndarray) else f"({len(arr)}, {image_height}, {image_width}, {image_channels})"

    def _min_max_median(arr):
        if isinstance(arr, np.ndarray):
            return float(arr.min()), float(arr.max()), float(np.median(arr))
        # sequence path: compute per-image extrema; approximate median from a sample
        mins = [float(im.min()) for im in arr]
        maxs = [float(im.max()) for im in arr]
        sample = arr[:min(64, len(arr))]
        med = float(np.median(np.concatenate([im.ravel() for im in sample]))) if sample else 0.0
        return float(np.min(mins)), float(np.max(maxs)), med

    # only print this if there is at least one image
    if _len_images(X_train) > 0:
        mn, mx, med = _min_max_median(X_train)
        TRex.log(f"X_train: {_shape_str(X_train)}, Y_train: {Y_train.shape} pixel min: {mn} max: {mx} median pixel: {med}")
    else:
        # generate some dummy data
        X_train = np.random.rand(1, image_height, image_width, image_channels)
        Y_train = np.random.randint(0, len(classes), size=(1,), dtype=int)
        TRex.log(f"Generated dummy data: X_train: {X_train.shape}, Y_train: {Y_train.shape} pixel min: {X_train.min()} max: {X_train.max()} median pixel: {np.median(X_train)}")

    if _len_images(X_test) > 0:
        mn, mx, med = _min_max_median(X_test)
        TRex.log(f"X_test: {_shape_str(X_test)}, Y_test: {Y_test.shape} pixel min: {mn} max: {mx} median pixel: {med}")
    else:
        X_test = np.random.rand(1, image_height, image_width, image_channels)
        Y_test = np.random.randint(0, len(classes), size=(1,), dtype=int)
        TRex.log(f"Generated dummy data: X_test: {X_test.shape}, Y_test: {Y_test.shape} pixel min: {X_test.min()} max: {X_test.max()} median pixel: {np.median(X_test)}")

    #train_data = TensorDataset(X_train, Y_train)
    #val_data = TensorDataset(X_test, Y_test)

    # Use TRexImageDataset + DataLoader (single-process) for stable threading behavior
    train_dataset = TRexImageDataset(X_train, Y_train, transform=transform, device=device)
    val_dataset = TRexImageDataset(X_test, Y_test, transform=None)

    pin_mem = (str(device) == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_mem
    )

    #train_loader = CustomDataLoader(X_train, Y_train, batch_size=batch_size, shuffle=True, transform=transform, device=device)
    #val_loader = CustomDataLoader(X_test, Y_test, batch_size=batch_size, device='cpu')

    settings["model"] = network_version
    settings["device"] = str(device)

    assert model is not None, "Model is not initialized."

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    mi = 0
    for i, c in zip(np.arange(len(classes)), classes):
        # Count max samples per class from integer labels
        mi = max(mi, int(np.sum(Y_train == c)))

    # Use DataLoader length (number of batches) for progress accounting
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
        per_class[i] = int(np.sum(Y_test == c))

        cvalues.append(per_class[i])
        if per_class[i] > m:
            m = per_class[i]

    if run_training:
        # Pass one-hot labels to callback; training uses integer labels
        callback = ValidationCallback(model, classes, X_test, Y_test_one_hot, max_epochs, filename, output_prefix, output_path, best_accuracy_worst_class, settings, device)

        TRex.log(f"# [init] weights per class {per_class}")
        TRex.log(f"# [training] data shapes: train={_shape_str(X_train)} {Y_train.shape} val={_shape_str(X_test)} {Y_test.shape} classes={classes}")

        if isinstance(X_train, np.ndarray):
            min_pixel_value = float(np.min(X_train))
            max_pixel_value = float(np.max(X_train))
        else:
            min_pixel_value = float(min(im.min() for im in X_train))
            max_pixel_value = float(max(im.max() for im in X_train))

        TRex.log(f"# [values] pixel values: min={min_pixel_value} max={max_pixel_value}")
        TRex.log(f"# [values] class values: min={np.min(Y_train)} max={np.max(Y_train)}")

        if do_save_training_images():
            TRex.log(f"# [training] saving training images to {output_path}_train_images.npz")
            try:
                np.savez(output_path+"_train_images.npz",
                        X_train=(X_train if isinstance(X_train, np.ndarray) else np.stack(X_train, axis=0)),
                        Y_train=Y_train,
                        classes=classes)
            except Exception as e:
                TRex.warn("Error saving training images: " + str(e))

        # Example training call
        try:
            train(model, train_loader, val_loader, criterion, optimizer, callback, scheduler=scheduler, device=device, settings=settings)
        except UserCancelException:
            print("Training cancelled by the user.")
        except UserSkipException:
            print("Training skipped by the user.")

        if callback.best_result["unique"] != -1:
            weights = callback.best_result["weights"]
            history_path = output_path+"_"+str(accumulation_step)+"_history.npz"
            TRex.log("saving histories at '"+history_path+"'")

            # load best results from the current run
            try:
                # Prefer new _dict.pth name; fall back to legacy .pth
                try:
                    TRex.log(f"Loading weights from {output_path+'_progress_dict.pth'} in step {accumulation_step}")
                    model, cp = load_model_from_file(output_path+'_progress_dict.pth', device=device, new_model = model)
                except Exception as e_load_new:
                    TRex.warn(f"Failed to load _progress_dict.pth: {e_load_new}; trying legacy _progress.pth")
                    TRex.log(f"Loading weights from {output_path+'_progress.pth'} in step {accumulation_step}")
                    model, cp = load_model_from_file(output_path+'_progress.pth', device=device, new_model = model)
                loaded_checkpoint = cp
                loaded_weights = TRex.VIWeights(
                    path = output_path+'_progress_dict.pth' if os.path.exists(output_path+'_progress_dict.pth') else output_path+'_progress.pth',
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

        callback = ValidationCallback(model, classes, X_test, Y_test_one_hot, max_epochs, filename, output_prefix, output_path, best_accuracy_worst_class, settings, device)
        if len(X_train) > 0:
            callback.evaluate(0, False)

        del callback
    
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
