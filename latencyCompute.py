import torch 
import torch.nn as nn
import torchvision

from utils import sram_traffic

arraySize = 64

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
        h = inDim_h + 2*p_h
        w = inDim_w + 2*p_w
        if g == 1:
            t,u = sram_traffic(dimension_rows=arraySize, dimension_cols=arraySize, 
                            ifmap_h=h, ifmap_w=w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=inC,strides=s_h, num_filt=outC)
            # print('Group=1 ',inDim_h, inC, outC, h, k_h, t, u)
            t = int(t)
        else:
            t,u = sram_traffic(dimension_rows=arraySize, dimension_cols=arraySize, 
                            ifmap_h=h, ifmap_w=w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=1,strides=s_h, num_filt=1)
            t = int(t)
            t = t*outC
            # print('Group > 1 ',inDim_h, inC, outC, h, k_h, t, u)

        self.time += t
    
    def clear(self):
        self.time = 0

models = torchvision.models.mobilenet_v2()


x = torch.rand([1, 3, 224, 224])
latency = ComputeLatency()
for layer in models.modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(latency)
print(models)
models(x)
print(latency.time)
# print(models)