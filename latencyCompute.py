import torch 
import torch.nn as nn
import torchvision

class ComputeLatency:
    def __init__(self):
        self.time = 0

    def __call__(self, module, module_in, module_out):
        inT = module_in[0]
        inDim = (inT.shape[2], inT.shape[3])
        inC = module.in_channels
        outC = module.out_channels
        k = module.kernel_size
        s = module.stride
        p = module.padding
        g = module.groups
        print(inDim, inC, outC, k, s, p, g)
        self.time += inC*outC
    
    def clear(self):
        self.time = 0

models = torchvision.models.resnet18()

x = torch.rand([1, 3, 224, 224])
latency = ComputeLatency()
def hookfn(module, inT, outT):
    print(module, inT[0].shape, type(inT), type(outT), outT.shape)

for layer in models.modules():
    if isinstance(layer, nn.Conv2d):
        print(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.groups)
        layer.register_forward_hook(latency)

models(x)
print(latency.time)
# print(models)