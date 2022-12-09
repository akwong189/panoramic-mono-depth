# Deployment

## Requirements

1. Build OpenCV with CUDA and GStreamer support, steps [here](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)
2. Install ONNX runtime for Tensorrt and CUDA support, can be built [here](https://onnxruntime.ai/docs/build/eps.html) or installing via pip `pip install onnxruntime-gpu`
3. Install requirements from requirements.txt `pip install -r requirements.txt`
4. Calibrate fisheye camera using the steps provided [here](https://docs.nvidia.com/vpi/sample_fisheye.html)
5. Create mapping array using `fish2pano.ipynb`, which will be downloaded
6. Using the ONNX model and the mapping, replace the top metrics

```python
MAP_FILE = "./map_full_900.npy"
ONNX_MODEL = "mobile_diode_pano_ssim_l1_sobel_edges_old_preprocess.onnx"
STORE_FRAMES = True
STORE_STATS = True
GUI = False
```