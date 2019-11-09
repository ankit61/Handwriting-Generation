import torch
import torch.nn as nn
import tensorboardX
import torchvision.models as models
from constants import STYLE_GENERATOR_ARCH, STYLE_VECTOR_SIZE
from BaseModule import BaseModule

class StyleGeneratorNetwork(BaseModule):
    def __init__(self, arch=STYLE_GENERATOR_ARCH):
        super(StyleGeneratorNetwork, self).__init__(debug=True)
        model_method_call = getattr(models, arch)
        self.model = model_method_call(pretrained=True, num_classes=STYLE_VECTOR_SIZE)

    def forward(self, x):
        return self.model(x)

    def introspect(self):
        return