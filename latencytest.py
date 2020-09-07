import torch
from models import *
from utils import *
import mobilenetv3fusedHybrid as mv3hybrid
import resnetfusedHybrid as resnetgyhbrid

x = torch.randn([1,3,224,224])
###########MobileNetV3##########
print("MobileNet-V3")
for size in ['small', 'large']:
    net = MobileNetV3(size, 1000)
    latb = getModelLatency(net, x)
    net = MobileNetV3Friendly(size, 1000)
    latf = getModelLatency(net, x)
    net = MobileNetV3Friendly2(size, 1000)
    latf2 = getModelLatency(net, x)
    net = mv3hybrid.MobileNetV3Friendly(size, 1000)
    lath = getModelLatency(net, x)
    net = mv3hybrid.MobileNetV3Friendly2(size, 1000)
    lath2 = getModelLatency(net, x)
    print(latb/latf, latb/latf2, latb/lath, latb/lath2)
#########
print('ResNet')
net = ResNet50(1000)
latb = getModelLatency(net, x)
net = ResNet50Friendly(1000)
latf = getModelLatency(net, x)
net = ResNet50Friendly2(1000)
latf2 = getModelLatency(net, x)
net = resnetgyhbrid.ResNet50Fused1(1000)
lath = getModelLatency(net, x)
net = resnetgyhbrid.ResNet50Fused2(1000)
lath2 = getModelLatency(net, x)
print(latb/latf, latb/latf2, latb/lath, latb/lath2)
####

