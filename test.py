from utils.CAM import *
from utils.Gard_CAM import *
from torchvision import models, transforms
import torch
from main import *
from model.net import *



# ------test CAM.py------
# model = DP1()

# pretrained = True
# img_path = 'input.jpg'
# oimg_path = 'game.jpg'
# # save_path = './CAM4test.jpg'
# label_path = './labels.json'
# checkpoint_path = 'params_4w.pth'

# cam = CAM(model, True , checkpoint_path,'cpu', img_path, oimg_path, label_path, 'cam', 'ocam')


# ------test Grad_CAM.py------
model = DP1()

pretrained = True
img_path = 'input.jpg'
oimg_path = 'game.jpg'
# save_path = './CAM4test.jpg'
label_path = './labels.json'
checkpoint_path = 'checkpoints\params_dp1_20w.pth'
finalconv_name = 'conv3'

cam = GradCAM(model, True , checkpoint_path, finalconv_name, 'cpu', img_path, oimg_path, label_path, 'cam/grad_cam/in_cam', 'cam/grad_cam/rgb_cam')




