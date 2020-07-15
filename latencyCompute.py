import math
import torch 
import torch.nn as nn
import torchvision

from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2 import MobileNetV2Friendly

from utils import sram_traffic

arraySize = 32

class ComputeLatency:
    def __init__(self):
        self.time = 0

    def __call__(self, module, module_in, module_out):
        global arraySize
        inT = module_in[0]
        inDim_h, inDim_w = (inT.shape[2], inT.shape[3])
        inC = module.in_channels
        outC = module.out_channels
        k_h, k_w = module.kernel_size
        s_h, s_w = module.stride
        p_h, p_w = module.padding
        g = module.groups
        inDim_h = inDim_h + 2*p_h
        inDim_w = inDim_w + 2*p_w
        if g == 1:
            t,u = sram_traffic(dimension_rows=arraySize, dimension_cols=arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=inC,strides=s_h, num_filt=outC)
            print('Group=1 ',inDim_h, inC, outC, k_h, t, u)
            t = int(t)
        else:
            if k_h == 1:
                num1Dconv = inDim_h * outC 
                numFolds = num1Dconv/arraySize
                oneFoldTime = arraySize + k_w
                num1DconvRow = inDim_h/arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
            elif k_w ==1 :
                num1Dconv = inDim_w * outC
                numFolds = num1Dconv/arraySize
                oneFoldTime = arraySize + k_h
                num1DconvRow = inDim_w/arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
            else:
                t,u = sram_traffic(dimension_rows=arraySize, dimension_cols=arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=1,strides=s_h, num_filt=1)
                t = int(t)
                t = t*outC
            
            print('Group > 1 ',inDim_h, inC, outC, k_h, t)

        self.time += t
    
    def clear(self):
        self.time = 0

def latency(model, x):
    hookfn = ComputeLatency()
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
    model(x)
    lat = hookfn.time
    hookfn.clear()
    return lat

x = torch.rand([1, 3, 224, 224])
model = MobileNetV2()
t1 = latency(model, x)
model = MobileNetV2Friendly()
# t2 = latency(model, x)
print(t1)#, t2, t1/t2)
