#!/usr/bin/env python
"""
convert_tf_model.py

!pip install tensorflow keras tf2onnx onnx2pytorch torch numpy

A script to convert a Keras model (stored in JSON format) to a frozen PyTorch model
via ONNX and TorchScript. Optionally loads model weights from a provided .npz file.

Dependencies:
    pip install tensorflow keras tf2onnx onnx2pytorch torch numpy

Usage:
    python convert_tf_model.py --json_model_path path/to/model.json [--weights_path path/to/weights.npz]
                               [--output_model_path path/to/output_model.pth]
                               [--image_width IMAGE_WIDTH --image_height IMAGE_HEIGHT --channels CHANNELS]
"""

import tensorflow
print("TensorFlow version:", tensorflow.__version__)

import argparse
import json
import keras
from keras.models import model_from_json
import tensorflow as tf
import tf2onnx
from onnx2pytorch import ConvertModel
import torch
import numpy as np
import json

def scaling_fn(x):
    """Custom scaling function to replace Lambda layers."""
    return x / 127.5 - 1.0

def convert_keras_to_torch(json_path, weights_path=None, image_width=None, image_height=None, channels=None, output_path=None):
    """
    Converts a Keras model stored in JSON format to a frozen PyTorch model via ONNX and TorchScript.
    
    Optionally loads model weights from a provided NumPy .npz file.
    
    Parameters:
      json_path (str): Path to the Keras JSON model file.
      weights_path (str, optional): Path to a NumPy (.npz) file containing model weights.
      image_width (int, optional): Input image width. If None, extracted from the model.
      image_height (int, optional): Input image height. If None, extracted from the model.
      channels (int, optional): Number of input channels. If None, extracted from the model.
      output_path (str, optional): If provided, the final TorchScript model is saved at this path.
      
    Returns:
      tuple: (torch.jit.ScriptModule, tuple) where the tuple is (image_width, image_height, channels).
      
    Raises:
      ValueError: If the input dimensions cannot be determined from the JSON file and are not provided.
    """
    # Load and update the model configuration.
    with open(json_path, "r") as f:
        model_dict = json.load(f)
    
    # If any input dimension is not provided, try to extract it from the JSON.
    if image_width is None or image_height is None or channels is None:
        found = False
        for layer in model_dict["config"]["layers"]:
            config = layer.get("config", {})
            if "batch_input_shape" in config:
                batch_input_shape = config["batch_input_shape"]
                # Determine the data format; default to channels_last.
                data_format = config.get("data_format", "channels_last")
                if data_format == "channels_last":
                    # batch_input_shape is [None, height, width, channels]
                    image_height = batch_input_shape[1]
                    image_width  = batch_input_shape[2]
                    channels     = batch_input_shape[3]
                else:
                    # For channels_first: [None, channels, height, width]
                    channels     = batch_input_shape[1]
                    image_height = batch_input_shape[2]
                    image_width  = batch_input_shape[3]
                found = True
                print(f"✅ Automatically detected input dimensions: width={image_width}, height={image_height}, channels={channels} in layer {layer['class_name']}")
                break
        
        if not found or image_width is None or image_height is None or channels is None:
            raise ValueError("⚠️ Could not determine input dimensions from the JSON file. "
                             "Please provide image_width, image_height, and channels explicitly.")
    
    # Replace Lambda layer function with our custom scaling function.
    found_input_layer = False
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "Lambda":
            layer["config"]["function"] = "scaling_fn"
            layer["config"]["function_type"] = "raw"
            layer["config"]["arguments"] = {}
        elif layer["class_name"] == "Input":
            found_input_layer = True

    if not found_input_layer:
        # we havent found an input layer, so we add one
        print("⚠️  No input layer found in the model. Please check the JSON file.")
        print("   Adding an input layer with the specified dimensions.")
        input_layer = {
            "class_name": "Input",
            "config": {
                "name": "input_1",
                "dtype": "float32",
                "batch_input_shape": [None, image_height, image_width, channels],
                "sparse": False,
                "ragged": False
            }
        }
        model_dict["config"]["layers"].insert(0, input_layer)
    
    new_json = json.dumps(model_dict)
    
    # Load and compile the Keras model.
    tf_model = model_from_json(new_json, custom_objects={
        "scaling_fn": scaling_fn,
        "Sequential": keras.Sequential,
        "Dense": keras.layers.Dense,
        "Dropout": keras.layers.Dropout,
        "Activation": keras.layers.Activation,
        "Cropping2D": keras.layers.Cropping2D,
        "Flatten": keras.layers.Flatten,
        "Convolution1D": keras.layers.Convolution1D,
        "Convolution2D": keras.layers.Convolution2D,
        "MaxPooling1D": keras.layers.MaxPooling1D,
        "MaxPooling2D": keras.layers.MaxPooling2D,
        "SpatialDropout2D": keras.layers.SpatialDropout2D,
        "Lambda": keras.layers.Lambda,
        "Input": keras.layers.Input,
        "BatchNormalization": keras.layers.BatchNormalization,
    })
    tf_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.legacy.Adam(),
                     metrics=['accuracy'])
    
    # Optionally, load model weights from a .npz file.
    if weights_path:
        with np.load(weights_path, allow_pickle=True) as npz:
            weights = npz['weights'].item()
            for i, layer in zip(range(len(tf_model.layers)), tf_model.layers):
                if i in weights:
                    layer.set_weights(weights[i])
        print(f"✅ Weights loaded from: {weights_path}")
    
    # Convert the Keras model to ONNX.
    spec = (tf.TensorSpec(tf_model.input.shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec)
    
    # Convert ONNX to PyTorch.
    pytorch_model = ConvertModel(onnx_model)
    pytorch_model.eval()
    
    # Prepare a dummy input and freeze the model using TorchScript.
    dummy_input = torch.randn(1, channels, image_height, image_width)
    try:
        scripted_model = torch.jit.script(pytorch_model)
        print("✅ Model scripted successfully.")
    except Exception as e:
        print("Scripting failed, falling back to tracing. Error:", e)
        scripted_model = torch.jit.trace(pytorch_model, dummy_input)
        print("✅ Model traced successfully.")
    
    # Optionally, save the final TorchScript model.
    if output_path:
        # Define metadata as a dictionary; here we serialize it to JSON.
        num_classes = model_dict["config"]["layers"][-1]["config"]["units"]
        metadata = {
            "input_shape": (image_width,image_height,channels),
            "model_type": "converted from Keras",
            "num_classes": num_classes,
            "epoch": None,
            "uniqueness": None
        }

        extra_files = {}
        extra_files["metadata"] = json.dumps(metadata)
        torch.jit.save(scripted_model, output_path, _extra_files=extra_files)
        print(f"TorchScript model with metadata saved at: {output_path}")

        files = {"metadata":""}
        loaded_model = torch.jit.load(output_path, _extra_files=files)

        metadata = json.loads(files["metadata"])
        # The metadata will be available in extra_files["metadata.json"]
        print(list(loaded_model.graph.inputs()))
        print("Loaded metadata:", files)
        print("converted to:", metadata)
    
    return scripted_model, (image_width, image_height, channels)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Keras model (JSON) to a frozen PyTorch model (TorchScript) via ONNX."
    )
    parser.add_argument("--json", required=True, help="Path to the Keras JSON model file.")
    parser.add_argument("--weights", default=None, help="Path to the weights .npz file (optional).")
    parser.add_argument("--output", default=None, help="Path to save the final TorchScript model (optional).")
    parser.add_argument("--width", type=int, default=None, help="Input image width (optional, auto-detected if not provided).")
    parser.add_argument("--height", type=int, default=None, help="Input image height (optional, auto-detected if not provided).")
    parser.add_argument("--channels", type=int, default=None, help="Number of input channels (optional, auto-detected if not provided).")
    
    args = parser.parse_args()
    
    try:
        model, shape = convert_keras_to_torch(
            json_path=args.json,
            weights_path=args.weights,
            image_width=args.width,
            image_height=args.height,
            channels=args.channels,
            output_path=args.output
        )
    except ValueError as ve:
        print(f"Error: {ve}")
        return
    
    print("Conversion successful. Detected model input shape (width, height, channels):", shape)
    
    # Test the converted model with a dummy input (format: batch, channels, height, width).
    dummy_input = torch.randn(1, shape[-1], shape[1], shape[0])
    with torch.no_grad():
        output = model(dummy_input)
    print("Test output:", output)

if __name__ == "__main__":
    main()