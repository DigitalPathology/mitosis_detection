import re
import torch
import torch.nn as nn

from torch.backends import cudnn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np
import os
import random

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



model_urls = \
    {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, param_7=7, param_512=512, param_4096=4096, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(param_7, param_7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=param_512 * param_7 * param_7, out_features=param_4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=param_4096, out_features=param_4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=param_4096, out_features=num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}





model = VGG(make_layers(cfgs['A'], batch_norm=False), param_7=7, param_512=512, param_4096=4096,init_weights=False)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained = True
progress   = True

if(pretrained):
    state_dict = load_state_dict_from_url(model_urls['vgg11'],progress=progress)
    model.load_state_dict(state_dict)


model = model.to(device)

if device.type == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True






num_classes = 2


#model.features = nn.Sequential(*(model.features[i] for i in range(11)))


# Add on classifier
model.classifier =  nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

print(model)
model = model.to('cuda')




# CUDA = torch.cuda.is_available()
# if CUDA:
#     model = model.cuda()
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.00001,momentum=0.9, nesterov=True, weight_decay=0.0005)



