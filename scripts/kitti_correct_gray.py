from time import sleep
import pandas as pd
import argparse
from rich.console import Console
from rich.progress import track
import cv2
from pathlib import Path
import os
from multiprocessing import Pool

import time
import skimage
import scipy
import numpy as np
from scipy.sparse.linalg import spsolve
from PIL import Image

def load_color_image(filename, **kwargs):
    img = cv2.imread(filename)
    return img

def load_depth_image(filename, **kwargs):
    depth = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return depth

console = Console()

# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso' solver 
#
#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.


def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    #print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    #print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1-knownValMask)) + imgDepthInput

    return output

depth_path = None
img_path = None

def perform_conversion(df):

    images = []
    depths = []

    # print(df)

    # for i in track(range(len(df.index))):
    for i in range(len(df.index)):
        start = time.time()
        image = df["images"][i]
        depth = df["depth"][i]

        new_depth_path = list(Path(depth).parts)
        new_depth_path[-3] = "correctedtruth"
        folder = "/".join(new_depth_path[:-1])
        new_depth_path = "/".join(new_depth_path)

        if os.path.exists(depth_path + new_depth_path):
            print(f"skipping - {new_depth_path} | {time.time() - start}s")
               
            images.append(image)
            depths.append(new_depth_path)
            continue
            
        os.makedirs(depth_path + folder, exist_ok=True)

        img = load_color_image(img_path + image)
        img = (img - img.min()) / (img.max() - img.min())

        d = load_depth_image(depth_path + depth)
        d = (d - d.min()) / (d.max() - d.min())

        result = fill_depth_colorization(img, np.squeeze(d))
        assert cv2.imwrite(depth_path + new_depth_path, result*255)

        images.append(image)
        depths.append(new_depth_path)
        print(f"done - {new_depth_path} | {time.time() - start}s")

    return {"images": images, "depth": depths}

def verify(path, depths):
    for step in track(range(len(depths))):
        if not os.path.exists(path + depths[step]):
            console.log(f"File not found: {path + depths[step]}", style="bold red")
            exit(1)

SIZE=15

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts image + depth to create contiguous depth images")
    parser.add_argument('file', help="File dataset generated from kitti_csv.py")
    parser.add_argument('write', help="File location to write a new csv")
    parser.add_argument('dpath', help="Path where the depth is located")
    parser.add_argument("ipath", help="Path to the images")

    args = parser.parse_args()

    df = pd.read_csv(args.file, index_col=0)

    l = len(df.index)
    print(l)
    parts = l // SIZE

    dfs = []
    for i in range(SIZE-1):
        dfs.append(df[parts*i:parts*(i+1)].reset_index(drop=True))
    dfs.append(df[parts*(SIZE-1):].reset_index(drop=True))

    print([len(d.index) for d in dfs])

    depth_path = args.dpath
    img_path = args.ipath

    with Pool(SIZE) as p:
        results = p.map(perform_conversion, dfs)

    total = {"images": [], "depth": []}

    for r in results:
        total["images"] += (r["images"])
        total["depth"] += (r["depth"])

    # verify(args.dpath, d)

    df = pd.DataFrame().from_dict(total)
    print(df)
    df.to_csv(args.write)
