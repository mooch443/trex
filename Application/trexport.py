import torch
import pickle
import cloudpickle
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision
import torch.backends.cudnn as cudnn

from utils.general import non_max_suppression
import numpy as np
import detectron2
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

from models.common import DetectMultiBackend
from utils.general import non_max_suppression# (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, 
import os

model_path = "Z:/work/shark/models/shark-cropped-0.712-loops-128-seg.pt"
output_path = model_path.replace('.pt', '')+"-windows.pth"
#model_path= "/Users/tristan/Downloads/shark-cropped-0.771-loops-64-seg.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t_model = DetectMultiBackend(model_path, device=device, dnn=False, fp16=False)

def predict_custom_yolo7_seg(t_model, device, image_size, offsets, im, conf_threshold = 0.25, iou_threshold = 0.1, mask_res = 56, max_det = 1000):
    def crop(masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        #print(n,h,w)
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        #print(x1,y1,x2,y2)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        #print(r)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        #print(c)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    if len(im.shape) < 4:
        im = im[np.newaxis, ...]

    print(image_size)
    if type(image_size) == np.ndarray or type(image_size) == list:
        image_size = int(image_size[0])

    if im.shape[1:] != (image_size,image_size,3):
    	print("Image shape unexpected, got ", im.shape, " expected (:",",",image_size,",",image_size,",3)")
    assert im.shape[1:] == (image_size,image_size,3)
    im0 = im.shape[1:]
    im = im.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    assert len(im.shape) == 4

    #dt = (Profile(), Profile(), Profile())

    #with dt[0]:
    im = torch.from_numpy(im).to(device)
    im = im.half() if t_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim



    def clip_coords(boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        clip_coords(coords, img0_shape)
        return coords

    def apply(pred, index, im, prot):
        meta = []
        indexes = []
        shapes = []
        ih, iw = im.shape[2:]

        c, mh, mw = prot.shape  # CHW

        downsampled_bboxes = pred[:, :4].clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = (pred[:, 6:] @ prot.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
        masks = crop(masks, downsampled_bboxes)

        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0).round()

        confs = pred[:, 4]
        clids = pred[:, 5]

        #for i, (det, proto) in enumerate(zip(pred, prot)):
            

        for j in range(len(masks)):
            x = masks[j]
            
            box = downsampled_bboxes[j]
            x0,y0,x1,y1 = box.cpu().numpy().astype(int)

            if x1-x0 > 0 and y1-y0 > 0 and x1+1 <= x.shape[1] and y1+1 <= x.shape[0] and x0 >= 0 and y0 >= 0:
                x = x[y0:y1+1, x0:x1+1]
                x = (T.Resize((mask_res,mask_res))(x[None])[0])# * 255).to(torch.uint8)
                shapes.append(x.cpu().numpy())
                indexes.append(index)
                #if not x.shape[0] == 0 and not x.shape[1] == 0:
                #    shapes.append(cv2.resize(x, (56,56)))
                meta.append(pred[j, :6].to(torch.float32).cpu().numpy())
            else:
                print("image empty :(")

            #meta.append(det[:, :6].to(torch.float32).cpu().numpy())
        assert len(meta) == len(indexes)
        return meta, indexes, shapes

    results = []
    shapes = []
    meta = []
    indexes = []

    #print("processing im.shape", im.shape)
    #print(offsets)
    offsets = np.reshape(offsets, (-1, 2)).astype(int)
    #print(len(pred))
    #assert len(offsets) == len(pred)
    pred = None
    prediction = None
    proto = None

    for i in range(len(im)):
        # Inference
        #local = (im[i:i+1].cpu().numpy() * 255).astype(np.uint8)[0, 0, ...]
        #print("local = ",local.shape, " ", local.dtype)
        #TRex.imshow("im"+str(len(im) - 1 - i), local)

        #with dt[1]:
        with torch.no_grad():
            pred, out = t_model(im[i][None], augment=None, visualize=None)
            proto = out[1]

            #print(proto.shape)

            # NMS
            #with dt[2]:
            #conf_thres = 0.1
            #iou_thres = 0.0
            
            pred = [a.to(device) for a in non_max_suppression(pred.cpu(), conf_threshold, iou_threshold, None, True, max_det=max_det, nm=32)]
            #print("nonmaxsupp proc", (pred[0]).int().cpu().numpy())
            _meta, _index, _shapes = apply(pred[0], index=i, im=im[i:i+1], prot = proto[0])
            #print("RESULT FOR ",i,"=",_meta)
        meta += _meta
        shapes += _shapes
        indexes += np.repeat(len(im) - 1 - i, len(_meta)).tolist()

    #print("meta: ", meta)
    #print("meta: ", np.concatenate(meta, axis=0))
    if len(meta) > 0:
        meta = np.concatenate(meta, axis=0, dtype=np.float32)
    else:
        meta = np.array([], dtype=np.float32)

    return np.array(shapes, dtype=np.float32).flatten(), meta.flatten(), np.array(indexes, dtype=int)

import models
import utils
cloudpickle.register_pickle_by_value(models)
cloudpickle.register_pickle_by_value(utils)
cloudpickle.register_pickle_by_value(detectron2)

print("Output to",output_path)

with open(output_path, "wb") as f:
	cloudpickle.dump({"model":t_model, "predict":predict_custom_yolo7_seg}, f)


# created using
#  git clone https://github.com/RizwanMunawar/yolov7-segmentation.git
#  cd yolov7-segmentation
#  $env:PYTHONPATH += "."
#  python ../trex/Application/trexport.py

# conda create -n trex_yolo python cmake ffmpeg cudatoolkit cudnn scikit-learn numpy pip 'tensorflow-gpu=2.6'
# python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# pip install opencv-python onnx==1.13 onnxruntime pyinstrument
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: win-64
# _tflow_select=2.1.0=gpu
# abseil-cpp=20210324.2=hd77b12b_0
# absl-py=1.3.0=py39haa95532_0
# aiohttp=3.8.3=py39h2bbff1b_0
# aiosignal=1.2.0=pyhd3eb1b0_0
# antlr4-python3-runtime=4.9.3=pypi_0
# anyio=3.5.0=py39haa95532_0
# argon2-cffi=21.3.0=pyhd3eb1b0_0
# argon2-cffi-bindings=21.2.0=py39h2bbff1b_0
# astor=0.8.1=py39haa95532_0
# asttokens=2.0.5=pyhd3eb1b0_0
# astunparse=1.6.3=py_0
# async-timeout=4.0.2=py39haa95532_0
# attrs=22.1.0=py39haa95532_0
# babel=2.11.0=py39haa95532_0
# backcall=0.2.0=pyhd3eb1b0_0
# beautifulsoup4=4.11.1=py39haa95532_0
# black=22.12.0=pypi_0
# blas=1.0=mkl
# bleach=4.1.0=pyhd3eb1b0_0
# blinker=1.4=py39haa95532_0
# bottleneck=1.3.5=py39h080aedc_0
# brotlipy=0.7.0=py39h2bbff1b_1003
# bzip2=1.0.8=he774522_0
# ca-certificates=2023.01.10=haa95532_0
# cachetools=4.2.2=pyhd3eb1b0_0
# certifi=2022.12.7=py39haa95532_0
# cffi=1.15.1=py39h2bbff1b_3
# chardet=4.0.0=pypi_0
# charset-normalizer=2.0.4=pyhd3eb1b0_0
# click=8.0.4=py39haa95532_0
# cloudpickle=2.2.1=pypi_0
# cmake=3.22.1=h9ad04ae_0
# colorama=0.4.6=py39haa95532_0
# comm=0.1.2=py39haa95532_0
# contourpy=1.0.7=pypi_0
# cryptography=38.0.4=py39h21b164f_0
# cuda-nvcc=12.0.140=0
# cudatoolkit=11.3.1=h59b6b97_2
# cudnn=8.2.1=cuda11.3_0
# cycler=0.10.0=pypi_0
# debugpy=1.5.1=py39hd77b12b_0
# decorator=5.1.1=pyhd3eb1b0_0
# defusedxml=0.7.1=pyhd3eb1b0_0
# detectron2=0.6=pypi_0
# entrypoints=0.4=py39haa95532_0
# executing=0.8.3=pyhd3eb1b0_0
# ffmpeg=4.2.2=he774522_0
# fftw=3.3.9=h2bbff1b_1
# flatbuffers=2.0.0=h6c2663c_0
# flit-core=3.6.0=pyhd3eb1b0_0
# fonttools=4.38.0=pypi_0
# freetype=2.12.1=ha860e81_0
# frozenlist=1.3.3=py39h2bbff1b_0
# fvcore=0.1.5.post20221221=pypi_0
# gast=0.4.0=pyhd3eb1b0_0
# giflib=5.2.1=h62dcd97_0
# git=2.34.1=haa95532_0
# google-auth=1.35.0=pypi_0
# google-auth-oauthlib=0.4.1=py_2
# google-pasta=0.2.0=pyhd3eb1b0_0
# grpcio=1.42.0=py39hc60d5dd_0
# h5py=3.7.0=py39h3de5c98_0
# hdf5=1.10.6=h1756f20_1
# hydra-core=1.3.1=pypi_0
# icc_rt=2022.1.0=h6049295_2
# icu=68.1=h6c2663c_0
# idna=2.10=pypi_0
# importlib-metadata=4.11.3=py39haa95532_0
# intel-openmp=2021.4.0=haa95532_3556
# iopath=0.1.9=pypi_0
# ipykernel=6.19.2=py39hd4e2768_0
# ipython=8.10.0=py39haa95532_0
# ipython_genutils=0.2.0=pyhd3eb1b0_1
# jedi=0.18.1=py39haa95532_1
# jinja2=3.1.2=py39haa95532_0
# joblib=1.1.1=py39haa95532_0
# jpeg=9e=h2bbff1b_0
# json5=0.9.6=pyhd3eb1b0_0
# jsonschema=4.17.3=py39haa95532_0
# jupyter_client=7.4.9=py39haa95532_0
# jupyter_core=5.2.0=py39haa95532_0
# jupyter_server=1.23.4=py39haa95532_0
# jupyterlab=3.5.3=py39haa95532_0
# jupyterlab_pygments=0.1.2=py_0
# jupyterlab_server=2.19.0=py39haa95532_0
# keras-preprocessing=1.1.2=pyhd3eb1b0_0
# kiwisolver=1.4.4=pypi_0
# lerc=3.0=hd77b12b_0
# libcurl=7.86.0=h86230a5_0
# libdeflate=1.8=h2bbff1b_5
# libiconv=1.16=h2bbff1b_2
# libpng=1.6.37=h2a8f88b_0
# libprotobuf=3.17.2=h23ce68f_1
# libsodium=1.0.18=h62dcd97_0
# libssh2=1.10.0=hcd4344a_0
# libtiff=4.5.0=h8a3f274_0
# libuv=1.40.0=he774522_0
# libwebp=1.2.4=h2bbff1b_0
# libwebp-base=1.2.4=h2bbff1b_0
# libxml2=2.9.14=h0ad7f3c_0
# libxslt=1.1.35=h2bbff1b_0
# lxml=4.9.1=py39h1985fb9_0
# lz4-c=1.9.4=h2bbff1b_0
# markdown=3.4.1=py39haa95532_0
# markupsafe=2.1.1=py39h2bbff1b_0
# matplotlib=3.6.3=pypi_0
# matplotlib-inline=0.1.6=py39haa95532_0
# mistune=0.8.4=py39h2bbff1b_1000
# mkl=2021.4.0=haa95532_640
# mkl-service=2.4.0=py39h2bbff1b_0
# mkl_fft=1.3.1=py39h277e83a_0
# mkl_random=1.2.2=py39hf11a4ad_0
# multidict=6.0.2=py39h2bbff1b_0
# mypy-extensions=0.4.3=pypi_0
# nbclassic=0.5.2=py39haa95532_0
# nbclient=0.5.13=py39haa95532_0
# nbconvert=6.5.4=py39haa95532_0
# nbformat=5.7.0=py39haa95532_0
# nest-asyncio=1.5.6=py39haa95532_0
# notebook=6.5.2=py39haa95532_0
# notebook-shim=0.2.2=py39haa95532_0
# numexpr=2.8.4=py39h5b0cc5e_0
# numpy=1.23.5=py39h3b20f71_0
# numpy-base=1.23.5=py39h4da318b_0
# oauthlib=3.2.1=py39haa95532_0
# omegaconf=2.3.0=pypi_0
# opencv-python=4.7.0.68=pypi_0
# openssl=1.1.1t=h2bbff1b_0
# opt_einsum=3.3.0=pyhd3eb1b0_1
# packaging=23.0=pypi_0
# pandas=1.5.2=py39hf11a4ad_0
# pandocfilters=1.5.0=pyhd3eb1b0_0
# parso=0.8.3=pyhd3eb1b0_0
# pathspec=0.10.3=pypi_0
# pickleshare=0.7.5=pyhd3eb1b0_1003
# pillow=9.4.0=pypi_0
# pip=22.3.1=py39haa95532_0
# platformdirs=2.6.2=pypi_0
# portalocker=2.7.0=pypi_0
# prometheus_client=0.14.1=py39haa95532_0
# prompt-toolkit=3.0.36=py39haa95532_0
# protobuf=3.17.2=py39hd77b12b_0
# psutil=5.9.0=py39h2bbff1b_0
# pure_eval=0.2.2=pyhd3eb1b0_0
# pyasn1=0.4.8=pyhd3eb1b0_0
# pyasn1-modules=0.2.8=py_0
# pycocotools=2.0.6=pypi_0
# pycparser=2.21=pyhd3eb1b0_0
# pygments=2.11.2=pyhd3eb1b0_0
# pyinstrument=4.4.0=pypi_0
# pyjwt=2.4.0=py39haa95532_0
# pyopenssl=22.0.0=pyhd3eb1b0_0
# pyparsing=2.4.7=pypi_0
# pyrsistent=0.18.0=py39h196d8e1_0
# pysocks=1.7.1=py39haa95532_0
# python=3.9.16=h6244533_0
# python-dateutil=2.8.2=pyhd3eb1b0_0
# python-dotenv=1.0.0=pypi_0
# python-fastjsonschema=2.16.2=py39haa95532_0
# python-flatbuffers=1.12=pyhd3eb1b0_0
# pytorch=1.12.1=py3.9_cuda11.3_cudnn8_0
# pytorch-mutex=1.0=cuda
# pytz=2022.7=py39haa95532_0
# pywin32=305=pypi_0
# pywinpty=2.0.2=py39h5da7b33_0
# pyyaml=6.0=pypi_0
# pyzmq=23.2.0=py39hd77b12b_0
# requests=2.28.1=py39haa95532_0
# requests-oauthlib=1.3.0=py_0
# requests-toolbelt=0.10.1=pypi_0
# roboflow=0.2.32=pypi_0
# rsa=4.7.2=pyhd3eb1b0_1
# scikit-learn=1.2.0=py39hd77b12b_0
# scipy=1.9.3=py39he11b74f_0
# seaborn=0.12.2=pypi_0
# send2trash=1.8.0=pyhd3eb1b0_1
# setuptools=65.6.3=py39haa95532_0
# six=1.16.0=pyhd3eb1b0_1
# snappy=1.1.9=h6c2663c_0
# sniffio=1.2.0=py39haa95532_1
# soupsieve=2.3.2.post1=py39haa95532_0
# sqlite=3.40.1=h2bbff1b_0
# stack_data=0.2.0=pyhd3eb1b0_0
# tabulate=0.9.0=pypi_0
# tensorboard=2.6.0=py_1
# tensorboard-data-server=0.6.1=py39haa95532_0
# tensorboard-plugin-wit=1.8.1=py39haa95532_0
# tensorflow=2.6.0=gpu_py39he88c5ba_0
# tensorflow-base=2.6.0=gpu_py39hb3da07e_0
# tensorflow-estimator=2.6.0=pyh7b7c402_0
# tensorflow-gpu=2.6.0=h17022bd_0
# termcolor=2.1.0=py39haa95532_0
# terminado=0.17.1=py39haa95532_0
# threadpoolctl=2.2.0=pyh0d69192_0
# tinycss2=1.2.1=py39haa95532_0
# tk=8.6.12=h2bbff1b_0
# tomli=2.0.1=py39haa95532_0
# torch=1.13.1+cu117=pypi_0
# torchaudio=0.13.1+cu117=pypi_0
# torchvision=0.14.1+cu117=pypi_0
# tornado=6.2=py39h2bbff1b_0
# tqdm=4.64.1=pypi_0
# traitlets=5.7.1=py39haa95532_0
# typing-extensions=4.4.0=py39haa95532_0
# typing_extensions=4.4.0=py39haa95532_0
# tzdata=2022g=h04d1e81_0
# urllib3=1.26.14=py39haa95532_0
# vc=14.2=h21ff451_1
# vs2015_runtime=14.27.29016=h5e58377_2
# wcwidth=0.2.5=pyhd3eb1b0_0
# webencodings=0.5.1=py39haa95532_1
# websocket-client=0.58.0=py39haa95532_4
# werkzeug=2.2.2=py39haa95532_0
# wget=3.2=pypi_0
# wheel=0.35.1=pyhd3eb1b0_0
# win_inet_pton=1.1.0=py39haa95532_0
# wincertstore=0.2=py39haa95532_2
# winpty=0.4.3=4
# wrapt=1.14.1=py39h2bbff1b_0
# xz=5.2.10=h8cc25b3_1
# yacs=0.1.8=pypi_0
# yarl=1.8.1=py39h2bbff1b_0
# zeromq=4.3.4=hd77b12b_0
# zipp=3.11.0=py39haa95532_0
# zlib=1.2.13=h8cc25b3_0
# zstd=1.5.2=h19a0ad4_0