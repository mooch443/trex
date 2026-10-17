# -*- coding: utf-8 -*-
"""
This module provides a function to load a checkpoint from a file and check its compatibility
with the current model configuration. It handles exported, JIT and standard PyTorch checkpoints.
It also includes a utility function to check the compatibility of checkpoint metadata
with the current model settings.
"""

import os
import json
import copy
import zipfile
import pickle
import torch
import torch.nn as nn
import TRex
import numpy as np
import gc

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
    image_width: int,
    image_height: int,
    image_channels: int,
    classes: list,
    metadata: dict,
    context: str = ""
):
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


class ExportedModel(nn.Module):
    """Inference adapter for an exported graph whose eval mode is fixed at export."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = False

    def forward(self, x):
        return self.module(x)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Exported models are inference-only; load their state_dict into a trainable model.")
        self.training = False
        return self


def load_checkpoint_from_file(file_path: str, device: str):
    """
    Loads a checkpoint from the specified file path.

    Accepts a .pt2 exported program, legacy TorchScript, or a .pth dictionary with:
      - A "model" field (for a complete model) and/or
      - A "state_dict" field (with optional "metadata").

    Returns a checkpoint dict (or a plain state dict). The caller checks metadata
    against its input dimensions and classes before applying the weights.
    """
    if not os.path.exists(file_path):
        raise Exception("Checkpoint file not found at " + file_path)
    if os.fspath(file_path).endswith(".pt2"):
        files = {"metadata": ""}
        program = torch.export.load(file_path, extra_files=files)
        metadata = None
        try:
            metadata = json.loads(files["metadata"])
        except Exception as e:
            TRex.warn("\t- Failed to load metadata from exported checkpoint: " + str(e))
        cp = {
            "model": ExportedModel(program.module()).to(device),
            "state_dict": program.state_dict,
            "metadata": metadata,
        }
        TRex.log(f"\t+ Loaded exported checkpoint from {file_path}.")
        return cp

    # TorchScript archives contain constants.pkl; weights-only archives do not.
    is_script = False
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as archive:
            is_script = any(name == "constants.pkl" or name.endswith("/constants.pkl")
                            for name in archive.namelist())

    if is_script:
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

    else:
        TRex.log(f"\t- Loading checkpoint with torch.load.")

        try:
            cp = torch.load(file_path, map_location=device, weights_only=True)
        except pickle.UnpicklingError:
            # Legacy dictionaries containing Python model objects need their classes.
            from torchvision import transforms
            from visual_identification_network_torch import (
                PermuteAxesWrapper, Normalize, V118_3, V110, V119, V200
            )
            # Register safe globals for torch.serialization.
            # Check if `torch.serialization.add_safe_globals` is available
            if hasattr(torch.serialization, "add_safe_globals"):
                # Register safe globals for torch.serialization.
                torch.serialization.add_safe_globals([PermuteAxesWrapper, Normalize, transforms.transforms.Normalize])
                torch.serialization.add_safe_globals([set])
                torch.serialization.add_safe_globals([V118_3, V110, V119, V200])
                torch.serialization.add_safe_globals([nn.Softmax, nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm, nn.ReLU,
                                                    nn.MaxPool2d, nn.Linear, nn.Dropout, nn.Dropout2d,
                                                    nn.LayerNorm, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                                                    nn.AvgPool2d, nn.MaxPool2d, nn.Flatten, nn.Sequential])
                torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt8DType, np.dtypes.Int64DType])
            else:
                # Log a warning or handle the absence of `add_safe_globals` gracefully
                TRex.warn("`torch.serialization.add_safe_globals` is not available in this version of PyTorch. Skipping safe globals registration.")

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
    
def save_pytorch_model_as_export(model, output_path, metadata):
    """
    Save an inference graph with metadata to .pt2, accepting dynamic NHWC batches.
    The CPU copy keeps the archive portable without changing the training model's
    device, buffers, or per-module train/eval modes. Metadata input_shape is W,H,C.
    """
    export_model = copy.deepcopy(model).cpu().eval()
    width, height, channels = metadata["input_shape"]
    # A sample batch of two avoids specializing the batch dimension to one.
    inputs = torch.zeros(2, height, width, channels, dtype=torch.float32)
    program = torch.export.export(
        export_model, (inputs,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
        strict=True,
    )
    torch.export.save(program, output_path, extra_files={"metadata": json.dumps(metadata)})
    TRex.log(f"Exported model with metadata saved at: {output_path}")
    return program

def clear_caches():
    device = TRex.choose_device()
    TRex.log(f"Clearing caches for {device}...")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        current_mem=torch.mps.current_allocated_memory()
        torch.mps.empty_cache()
        TRex.log(f"Current memory: {current_mem/1024/1024}MB -> {torch.mps.current_allocated_memory()/1024/1024}MB")
    else:
        TRex.log(f"No cache to clear {device}")

    gc.collect()

def asarray(obj, copy=None, dtype=None):
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
        return np.asarray(obj, copy=copy, dtype=dtype)
    else:
        return np.array(obj, copy=copy if copy is not None else True, dtype=dtype)  # Default copy=True for older numpy versions
    
# ---- helpers to support ndarray or list-of-ndarrays -----------------
def _as_batched_np(X, idx):
    """Return a (B,H,W,C) ndarray given either a big ndarray or a list/tuple of ndarrays.
    If X is an ndarray, this uses NumPy slicing (usually a view, no host copy).
    If X is a sequence of ndarrays, we gather and stack once (one host copy for that batch).
    """
    if isinstance(X, np.ndarray):
        return X[idx]
    # sequence path
    if isinstance(idx, slice):
        # Some pybind-backed sequences may not support slice objects; fall back to gather
        try:
            imgs = X[idx]
        except Exception:
            start, stop, step = idx.indices(len(X))
            imgs = [X[i] for i in range(start, stop, step)]
    else:
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()
        imgs = [X[i] for i in idx]
    return np.stack(imgs, axis=0)

def _first_shape(X):
    """Return (H,W,C) from either (N,H,W,C) ndarray or list of (H,W,C) ndarrays."""
    if isinstance(X, np.ndarray):
        assert X.ndim == 4, "Invalid image shape"
        return X.shape[1], X.shape[2], X.shape[3]
    else:
        first = X[0]
        assert isinstance(first, np.ndarray) and first.ndim == 3, "Expect list/tuple of HxWxC ndarrays"
        return first.shape

class UserCancelException(Exception):
    """Raised when user clicks cancel"""
    pass

class UserSkipException(Exception):
    """Raised when user clicks cancel"""
    pass
