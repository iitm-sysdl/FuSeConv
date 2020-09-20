import torch
import time
from models import *
from utils import *

x = torch.randn([1,3,224,224])
mode =  'analytical'
arraySize = 64

otherConv = [] 
pointConv = [] 
depthConv = []
linear = []
supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
for net in supernet:
    a, b, c, d = getModelLatencyBreakdown(net, x, mode, arraySize)
    otherConv.append(a)
    pointConv.append(b)
    depthConv.append(c)
    linear.append(d)

print(otherConv, pointConv, depthConv, linear)

otherConv = [] 
pointConv = [] 
depthConv = []
linear = []
supernet = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
for net in supernet:
    a, b, c, d = getModelLatencyBreakdown(net, x, mode, arraySize)
    otherConv.append(a)
    pointConv.append(b)
    depthConv.append(c)
    linear.append(d)
    

print(otherConv, pointConv, depthConv, linear)

otherConv = [] 
pointConv = [] 
depthConv = []
linear = []
supernet = [MobileNetV1Friendly2(1000), MobileNetV2Friendly2(1000), MnasNetFriendly2(1000), MobileNetV3Friendly2('small', 1000), MobileNetV3Friendly2('large', 1000)]
for net in supernet:
    a, b, c, d = getModelLatencyBreakdown(net, x, mode, arraySize)
    otherConv.append(a)
    pointConv.append(b)
    depthConv.append(c)
    linear.append(d)

print(otherConv, pointConv, depthConv, linear)