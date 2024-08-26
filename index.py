import torch
from matplotlib.pyplot import imshow  
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch import topk
from torch.nn import functional as F
import numpy as np 
import skimage.transform

image = Image.open("./cat.jpg")
imshow(image)

print(torch.cuda.is_available())