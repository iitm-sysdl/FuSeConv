import torch
import time
from models import *
from utils import *

supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
supernetF = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]

x = torch.randn([1,3,224,224])
mode = 'analytical'
arraySize = [32, 64, 128, 256]
speedup = []
for i,j in enumerate(supernet):
    net1 = supernet[i]
    net2 = supernetF[i]
    pernetSpeed = []
    for s in arraySize:
        lat1 = getModelLatency(net1, x, mode, s)
        lat2 = getModelLatency(net2, x, mode, s)
        pernetSpeed.append(lat1/lat2)
    speedup.append(pernetSpeed)

print(speedup)

supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
supernetF = [MobileNetV1Friendly2(1000), MobileNetV2Friendly2(1000), MnasNetFriendly2(1000), MobileNetV3Friendly2('small', 1000), MobileNetV3Friendly2('large', 1000)]

x = torch.randn([1,3,224,224])
mode = 'analytical'
arraySize = [32, 64, 128, 256]
speedup = []
for i,j in enumerate(supernet):
    net1 = supernet[i]
    net2 = supernetF[i]
    pernetSpeed = []
    for s in arraySize:
        lat1 = getModelLatency(net1, x, mode, s)
        lat2 = getModelLatency(net2, x, mode, s)
        pernetSpeed.append(lat1/lat2)
    speedup.append(pernetSpeed)

print(speedup)