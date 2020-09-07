import torch
import time
from models import *
from utils import *

import mobilenetv3fusedHybrid as mv3hybrid
import resnetfusedHybrid as resnetgyhbrid
import mobilenetv2fusedHybrid as mv2hybrid

x = torch.randn([1,3,224,224])
mode = 'analytical'
# model ='scale-sim'
arraySize = 16
start_time = time.time()
###########MobileNetV3##########
print("MobileNet-V3")
for size in ['small', 'large']:
    net = MobileNetV3(size, 1000)
    latb = getModelLatency(net, x, mode, arraySize)
    net = MobileNetV3Friendly(size, 1000)
    latf = getModelLatency(net, x, mode, arraySize)
    net = MobileNetV3Friendly2(size, 1000)
    latf2 = getModelLatency(net, x, mode, arraySize)
    net = mv3hybrid.MobileNetV3Friendly(size, 1000)
    lath = getModelLatency(net, x, mode, arraySize)
    net = mv3hybrid.MobileNetV3Friendly2(size, 1000)
    lath2 = getModelLatency(net, x, mode, arraySize)
    print(latb/latf, latb/latf2, latb/lath, latb/lath2)
#########
print("--- %s seconds ---" % (time.time() - start_time))
print('ResNet')
net = ResNet50(1000)
latb = getModelLatency(net, x, mode, arraySize)
net = ResNet50Friendly(1000)
latf = getModelLatency(net, x, mode, arraySize)
net = ResNet50Friendly2(1000)
latf2 = getModelLatency(net, x, mode, arraySize)
net = resnetgyhbrid.ResNet50Fused1(1000)
lath = getModelLatency(net, x, mode, arraySize)
net = resnetgyhbrid.ResNet50Fused2(1000)
lath2 = getModelLatency(net, x, mode, arraySize)
print(latb/latf, latb/latf2, latb/lath, latb/lath2)
####
print("--- %s seconds ---" % (time.time() - start_time))
print('MobileNet-V2')
net = MobileNetV2(1000)
latb = getModelLatency(net, x, mode, arraySize)
net = MobileNetV2Friendly(1000)
latf = getModelLatency(net, x, mode, arraySize)
net = MobileNetV2Friendly2(1000)
latf2 = getModelLatency(net, x, mode, arraySize)
net = mv2hybrid.MobileNetV2Friendly(1000)
lath = getModelLatency(net, x, mode, arraySize)
net = mv2hybrid.MobileNetV2Friendly2(1000)
lath2 = getModelLatency(net, x, mode, arraySize)
print(latb/latf, latb/latf2, latb/lath, latb/lath2)
#####
print("--- %s seconds ---" % (time.time() - start_time))
print('MobileNet-V1')
net = MobileNetV1(1000)
latb = getModelLatency(net, x, mode, arraySize)
net = MobileNetV1Friendly(1000)
latf = getModelLatency(net, x, mode, arraySize)
net = MobileNetV1Friendly2(1000)
latf2 = getModelLatency(net, x, mode, arraySize)
# net = mv2hybrid.MobileNetV2Friendly(1000)
# lath = getModelLatency(net, x, mode, arraySize)
# net = mv2hybrid.MobileNetV2Friendly2(1000)
# lath2 = getModelLatency(net, x, mode, arraySize)
print(latb/latf, latb/latf2)# latb/lath, latb/lath2)


