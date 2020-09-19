import torch
import time
from models import *
from utils import *

network=['MobileNetV1', 'MobileNetV2', 'MnasNet', 'MobileNet-V3\nSmall', 'MobileNet-V3\nLarge']
speedup = []
speedup2 = []
x = torch.randn([1,3,224,224])
# mode = 'scale-sim'
mode =  'analytical'
arraySize = 64

otherConv = [] 
pointConv = [] 
depthConv = []
supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
for net in supernet:
    a, b, c = getModelLatencyBreakdown(net, x, mode='analytical', arraySize=8)
    otherConv.append(a)
    pointConv.append(b)
    depthConv.append(c)

print(otherConv, pointConv, depthConv)

otherConv = [] 
pointConv = [] 
depthConv = []
supernet = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
for net in supernet:
    a, b, c = getModelLatencyBreakdown(net, x, mode='analytical', arraySize=8)
    otherConv.append(a)
    pointConv.append(b)
    depthConv.append(c)

print(otherConv, pointConv, depthConv)