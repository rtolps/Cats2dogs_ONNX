import torch.onnx
from torch import nn

from utils import *
from dataset import ImageFolder
from networks import *

class Model(object) :
    def __init__(self):
        super().__init__()
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, 
                                      ngf=16, n_blocks=4, img_size=256, light=True).to('cpu') #origional ngf was 64

    def forward(self, x):
        out = self.genA2B(x)
        out = nn.functional.interpolate(out, scale_factor=2, 
                                        mode='bilinear', align_corners=True)
        out = torch.nn.functional.softmax(out, dim=1)
        return out
model = Model()
params = torch.load('/content/Cats2dogs_ONNX/results/cat2dog/model/cat2dog_params_0000100.pt') #guessing what step is equal too
model.genA2B.load_state_dict(params['genA2B'])
model.genA2B.eval()
random_input = torch.randn(3, 3, 256, 256, dtype=torch.float32)
# you can add however many inputs your model or task requires
 
input_names = ["real_A"]
output_names = ["fake_A2B"]
 
torch.onnx.export(model.genA2B, random_input, 'model.onnx', verbose=False, 
                  input_names=input_names, output_names=output_names, 
                  opset_version=11)
