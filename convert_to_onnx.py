import torch.onnx
from torch import nn

import config
from dataset import ImageFolder
from networks import *

class Model(object, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
    def __init__(self):
        super().__init__()
        self.n_res = args.n_res
        
    def forward(self, x):
        out = self.n_res(x)
        out = nn.functional.interpolate(out, scale_factor=2, 
                                        mode='bilinear', align_corners=True)
        out = torch.nn.functional.softmax(out, dim=1)
        return out