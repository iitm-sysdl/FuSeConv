import os
import torch
import random
import argparse
import torchvision

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *
from cifarmodels import *

def main():
    meta = open('modelStats.txt', "w")
    s = 'Network , numClasses  , Type , FLOPS , Params , Latency 4x4 , Latency 8x8 , Latency 16x16 , Latency 32x32'  
    for network in ['ResNet', 'MobileNet', 'EfficientNet', 'SqueezeNet']:
        for numClasses in [10, 100]:
            for baseline in ['Baseline', 'Friendly']:
                if baseline == 'Baseline':
                    if network == 'ResNet':
                        net = ResNet50(numClasses)
                    elif network == 'MobileNet':
                        net = MobileNetV2(numClasses)
                    elif network == 'EfficientNet':
                        net = EfficientNetB0(numClasses)
                    elif network == 'SqueezeNet':
                        net = SqueezeNet(numClasses)
                else:
                    if network == 'ResNet':
                        net = ResNet50Friendly(numClasses)
                    elif network == 'MobileNet':
                        net = MobileNetV2Friendly(numClasses)
                    elif network == 'EfficientNet':
                        net = EfficientNetB0Friendly(numClasses)
                    elif network == 'SqueezeNet':
                        net = SqueezeNetFriendly(numClasses)
                
                x = torch.rand([1,3,32,32])
                flops, params = getModelProp(net, x)
                latency = []
                for size in [4, 8, 16, 32]:
                    latency.append(getModelLatency(net, x, arraySize=size))
                latency = ' , '.join(str(i) for i in latency)
                s = network + ' , ' + str(numClasses) + ' , ' + baseline + ' , ' + str(flops) + ' , ' + str(params) + ' , ' + latency + ' , ' + '\n'  
                meta.write(s)    
    meta.close()
    
if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    main()
