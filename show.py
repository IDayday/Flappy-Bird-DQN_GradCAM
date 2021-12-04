import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from CAM import *

img_path = 'input.jpg'
img_pil = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
print(img_pil.shape)
img_ex = np.expand_dims(img_pil,0).repeat(4,axis=0)
print(img_ex.shape)
img_ex = preprocess(img_ex).unsqueeze(0).reshape(1,4,80,80)
print(img_ex.shape)