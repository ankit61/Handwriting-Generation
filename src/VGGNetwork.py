import torch
import torch.nn as nn
import tensorboardX
import torchvision.models as models
from BaseModule import BaseModule

class VGGNetwork(BaseModule):

    def __init__(self, features, num_classes):
        self.vgg = models.VGG(features, num_classes=num_classes)

    def introspect(self):
        return