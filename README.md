# [Monocular Panoramic Depth Perception Model and Framework for micro-architectures](tbd) By Adley Wong

## Abstract

Depth perception has become a heavily researched area as companies and researchers are striving towards the development of self-driving cars. Self-driving cars rely on perceiving the surrounding area, which heavily depends on technology capable of providing the system with depth perception capabilities. In this paper, we explore developing a single camera (monocular) depth prediction model that is trained on panoramic depth images. Our model makes novel use of transfer learning efficient encoder models, pre-training on a larger dataset of flat depth images, and optimizing the model for use with a Jetson Nano. Additionally, we present a training and optimization framework to make developing and testing new monocular depth perception models easier and faster. While the model failed to achieve a high frame rate, the framework and models developed are a promising starting place for future work.

## Setup

1. Install tensorflow-gpu, which can be easily done using anaconda
2. Download and extract datasets [Pano3D](https://vcl3d.github.io/Pano3D/download/) and [Diode](https://diode-dataset.org), more can be installed but will need to supply splits
3. Install requirements.txt `pip install -r requirements.txt`

## Framework

### Training

```sh
usage: panodepth.py train [-h] -d {pano,kitti,diode,nyu} [-m {efficient,mobile,opt,scene,vgg,shuffle,mobilev3}] -o OUTPUT [-g {0,1,2,3}] [-s SEED] [-p PATH] [-l LOAD] [-e EPOCHS] [-lr RATE] [--cpu] [--summary]
                          [--loss {ssim,l1,berhu,sobel,edges,smooth} [{ssim,l1,berhu,sobel,edges,smooth} ...]] [--verbose [0-2]]

optional arguments:
  -h, --help            show this help message and exit
  -d {pano,kitti,diode,nyu}, --dataset {pano,kitti,diode,nyu}
                        select dataset to train
  -m {efficient,mobile,opt,scene,vgg,shuffle,mobilev3}, --model {efficient,mobile,opt,scene,vgg,shuffle,mobilev3}
                        model to train dataset on
  -o OUTPUT, --output OUTPUT
                        output .h5 file name to results
  -g {0,1,2,3}, --gpu {0,1,2,3}
                        set gpu to train on
  -s SEED, --seed SEED  set seed for training
  -p PATH, --path PATH  path to dataset
  -l LOAD, --load LOAD  load .h5 file for retraining and/or metric calculations
  -e EPOCHS, --epochs EPOCHS
                        set the number of epochs to train for
  -lr RATE, --rate RATE
                        set learning rate
  --cpu                 Use CPU instead of GPU
  --summary             Display the model summary
  --loss {ssim,l1,berhu,sobel,edges,smooth} [{ssim,l1,berhu,sobel,edges,smooth} ...]
                        set loss function to use for training
  --verbose [0-2]       set verbosity of training
```

### Metrics

```sh
usage: panodepth.py metrics [-h] -m MODEL [--cpu] [-g {0,1,2,3}]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        .h5 or .onnx model
  --cpu                 Use CPU instead of GPU
  -g {0,1,2,3}, --gpu {0,1,2,3}
                        set gpu to train on
```

### Optimization

```sh
usage: panodepth.py optimize [-h] -m MODEL [-q] [-l {ssim,l1,berhu,sobel,edges,smooth} [{ssim,l1,berhu,sobel,edges,smooth} ...]] [--cpu] [-g {0,1,2,3}]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        .h5 or .onnx model
  -q, --quantize        quantize model
  -l {ssim,l1,berhu,sobel,edges,smooth} [{ssim,l1,berhu,sobel,edges,smooth} ...], --loss {ssim,l1,berhu,sobel,edges,smooth} [{ssim,l1,berhu,sobel,edges,smooth} ...]
                        set loss function to use for training
  --cpu                 Use CPU instead of GPU
  -g {0,1,2,3}, --gpu {0,1,2,3}
                        set gpu to train on
```
