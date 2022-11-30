from models.TCSVT import DownSampling, UpSampling, Scene_Understanding
from models import efficientnet, mobilenet, optimizedmobilenet, vgg, mobilenetv3, wnet
from loss import *

# !!! ADD NEW MODEL CUSTOM FUNCTIONS FOR LOADING .h5 MODELS HERE !!!
custom_func = {
    "DownSampling": DownSampling,
    "UpSampling": UpSampling,
    "Scene_Understanding": Scene_Understanding,
}
# !!! ADD NEW MODEL CUSTOM FUNCTIONS FOR LOADING .h5 MODELS HERE !!!

# !!! ADD NEW MODEL REFERENCES HERE !!!
custom_models = {
    "efficient": efficientnet.EfficientUNet,
    "mobile": mobilenet.MobileNet,
    "mobilev3": mobilenetv3.MobileNetv3,
    "opt": optimizedmobilenet.OptimizedUNet,
    "scene": optimizedmobilenet.OptimizedUNet_Scene,
    "vgg": vgg.VGG,
    "shuffle": wnet.wnet
}
# !!! ADD NEW MODEL REFERENCES HERE !!!

# !!! ADD LOSS FUNCTION MODULE REFERENCES HERE !!!
loss2func = {
    "ssim": ssim_loss,
    "l1": l1_loss,
    "berhu": berhu_loss,
    "sobel": sobel_loss,
    "smooth": depth_smoothness_loss,
    "edges": edge_loss
}
# !!! ADD LOSS FUNCTION MODULE REFERENCES HERE !!!