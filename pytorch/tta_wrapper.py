#coding=utf8

import torch
import numpy as np
from PIL import Image
from torchvision import  models, transforms


class TTA_ModelWrapper(torch.nn.Module):
    """A simple TTA wrapper for pytorch model.

    Args:
        model: a pytorch model with a forward method.

    """

    def __init__(self,model):
        super(TTA_ModelWrapper, self).__init__()
        self.model = model

    """

    Args:
        x: the data to get predictions for.
    """

    def forward(self,x):
        results=[]
        #for idx,item in enumerate(x):
        #    output = self.model(item)
        #    print output
        return self.model(x) 
        #return results