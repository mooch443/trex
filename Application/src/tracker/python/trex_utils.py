# -*- coding: utf-8 -*-
"""
This module provides a function to load a checkpoint from a file and check its compatibility
with the current model configuration. It handles both JIT and standard PyTorch checkpoints.
It also includes a utility function to check the compatibility of checkpoint metadata
with the current model settings.
"""

import os
import json
import torch
import torch.nn as nn
import TRex
from torchvision import transforms
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
        # Check if `torch.serialization.add_safe_globals` is available
        if hasattr(torch.serialization, "add_safe_globals"):
            # Register safe globals for torch.serialization.
            torch.serialization.add_safe_globals([PermuteAxesWrapper, Normalize, transforms.transforms.Normalize])
            torch.serialization.add_safe_globals([set])
            torch.serialization.add_safe_globals([V118_3, V110, V119, V200])
            torch.serialization.add_safe_globals([nn.Softmax, nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
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
    print("JIT Loaded metadata:", {key:str(loaded_metadata[key])[:100] for key in loaded_metadata})
    
    return scripted_model

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
