.. include:: names.rst

.. toctree::
   :maxdepth: 3
   :numbered:

Training your own model
=======================

Introduction
------------
In this tutorial, we will walk you through the process of training your very own YOLOv11 model to be integrated into |trex|. Whether you’re a deep learning beginner or a seasoned practitioner, this guide provides a friendly and practical roadmap to get your model up and running.

Setting Up the Environment
---------------------------
For this tutorial, we recommend using Google Colab, which provides a pre-configured environment for training models along with free access to GPU resources. Using Colab allows you to get started quickly without the need to install dependencies locally.

There are many example notebooks available online to guide you through the process. You can view one provided by Ultralytics here: `Train YOLOv11 on a Custom Dataset <https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb>`_.

Data Preparation
----------------
A YOLO dataset is typically composed of images and corresponding annotation files. Each annotation file contains the object class and normalized bounding box coordinates (center_x, center_y, width, height) for the objects in the image.

Organize your dataset into the following directory structure:

.. code-block:: none

   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/

Ensure that your annotations follow the standard YOLO format. If you’re new to machine learning and object detection, consider reviewing this example notebook for a practical introduction: `Train YOLOv11 on a Custom Dataset <https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb>`_ and `Training custom datasets with Ultralytics YOLOv8 in Google Colab <https://www.ultralytics.com/blog/training-custom-datasets-with-ultralytics-yolov8-in-google-colab>`_.

Of course, don't forget to update your training script to point to your data as demonstrated there. 😊

Configuring YOLOv11
-------------------
Customize your YOLOv11 configuration to suit your training needs. Open your configuration file (typically a .yaml or .cfg file) and adjust the following as needed:

- :param:`batch` – The number of images to process per training iteration.
- :param:`imgsz` – The dimensions for your input images.
- :param:`epochs` – The number of training epochs. Longer training takes longer (obviously), but this can help fine-tune on your data. With diminishing returns.

These adjustments ensure that the model architecture and training parameters match your dataset and computational resources.

Troubleshooting and Tips
------------------------
Here are a few tips to help you get the best results:

- **Common Issues:** If you experience slow training or memory errors, consider reducing the :param:`batch` or :param:`imgsz`. You can also try using a smaller model variant (e.g., YOLO11n instead of YOLO11x) to speed up training. Check out the `ultralytics website <https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes>`_ on this topic.
- **Data Augmentation:** Use data augmentation techniques (e.g., rotations, flips, scaling) to increase the diversity of your training data and improve model robustness. But be careful: for pose models containing keypoints like left-/right-arm be sure to disable any rotation / mirroring!
- **Is my model any good?:** To evaluate your model's performance, consider using metrics such as mAP (mean Average Precision) and F1 score. An mAP of above 90% is usually a good sign. Visualizing predictions on a validation set can also provide insights into its accuracy.