# FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays  [[Paper](https://surya00060.github.io/files/FuSeConv_DATE_2021.pdf)][[Short Slides](https://slides.com/vinodganesan/fuseconv_date_2021-9d8347/fullscreen?token=twJbUI6C)][[Full Slides](https://slides.com/vinodganesan/fuseconv_date_2021/fullscreen?token=0vfMX47V)][[Video]()]

```BibTex
```

## Results

### MobileNet-V1

|           Network          | ImageNet  Accuracy | FLOPS (M) | Params (M) | Speedup |
|:--------------------------:|:------------------:|:---------:|:----------:|:-------:|
|   MobileNet V1 Baseline    |        70.60       |    589    |    4.23    |    1x   |
|   MobileNet V1 Full FuSe   |        72.86       |    1122   |    7.36    |   4.1x  |
|   MobileNet V1 Half FuSe   |        72.00       |    573    |    4.20    |  6.76x  |
| MobileNet V1 50% Full FuSe |        72.42       |    764    |    4.35    |   2.3x  |
| MobileNet V1 50% Half FuSe |        71.77       |    578    |    4.22    |  2.36x  |

### MobileNet-V2

|           Network          | ImageNet  Accuracy | FLOPS (M) | Params (M) | Speedup |
|:--------------------------:|:------------------:|:---------:|:----------:|:-------:|
|   MobileNet V2 Baseline    |        72.00       |    315    |    3.50    |    1x   |
|   MobileNet V2 Full FuSe   |        72.49       |    430    |    4.46    |   5.1x  |
|   MobileNet V2 Half FuSe   |        70.80       |    300    |    3.46    |  7.23x  |
| MobileNet V2 50% Full FuSe |        72.11       |    361    |    3.61    |   2.0x  |
| MobileNet V2 50% Half FuSe |        71.98       |    305    |    3.49    |   2.1x  |

### MNasNet-B1

|          Network         | ImageNet  Accuracy | FLOPS (M) | Params (M) | Speedup |
|:------------------------:|:------------------:|:---------:|:----------:|:-------:|
|   MnasNet B1 Baseline    |        73.50       |    325    |    4.38    |    1x   |
|   MnasNet B1 Full FuSe   |        73.16       |    440    |    5.66    |  5.06x  |
|   MnasNet B1 Half FuSe   |        71.48       |    305    |    4.25    |  7.15x  |
| MnasNet B1 50% Full FuSe |        73.52       |    361    |    4.47    |  1.88x  |
| MnasNet B1 50% Half FuSe |        72.61       |    312    |    4.35    |  1.97x  |

### MobileNet-V3 Small

|              Network             | ImageNet  Accuracy | FLOPS (M) | Params (M) | Speedup |
|:--------------------------------:|:------------------:|:---------:|:----------:|:-------:|
|    MobileNet V3 Small Baseline   |        67.40       |     66    |    2.93    |    1x   |
|   MobileNet V3 Small Full FuSe   |        67.17       |     84    |    4.44    |  3.02x  |
|   MobileNet V3 Small Half FuSe   |        64.55       |     61    |    2.89    |  4.16x  |
| MobileNet V3 Small 50% Full FuSe |        67.91       |     73    |    3.18    |   1.6x  |
| MobileNet V3 Small 50% Half FuSe |        66.90       |     63    |    2.92    |  1.68x  |

### MobileNet-V3 Large

|              Network             | ImageNet  Accuracy | FLOPS (M) | Params (M) | Speedup |
|:--------------------------------:|:------------------:|:---------:|:----------:|:-------:|
|    MobileNet V3 Large Baseline   |        75.20       |    238    |    5.47    |    1x   |
|   MobileNet V3 Large Full FuSe   |        74.40       |    322    |    10.57   |  3.61x  |
|   MobileNet V3 Large Half FuSe   |        73.02       |    225    |    5.40    |  5.45x  |
| MobileNet V3 Large 50% Full FuSe |        74.50       |    264    |    5.57    |  1.76x  |
| MobileNet V3 Large 50% Half FuSe |        73.80       |    230    |    5.46    |  1.83x  |

## PyTorch Model Codes

The code for Full/Half DNN varaints can be found in ```models``` directory.
* MobileNet V1
* MobileNet V2
* MobileNet V3
* MnasNet-B1
* ResNet
* VGG
* SqueezeNet

The hybrid or 50% varaints can be found in 

## Train on ImageNet

## Train on CIFAR

CIFAR images 32x32 are resized or upscaled to 224x224 and used to train the models.  

```
python traincifar224.py -D Dataset -N Network -n NameoftheRun -v Variant

--resume : For Resuming the Run
--baseline: For running the baseline model

Options: 
Dataset = CIFAR10, CIFAR100  
Network = ResNet, VGG, SqueezeNet, MobileNetV1, MobileNetV2, MobileNetV3S, MobileNetV3L, MnasNet 
Variant = friendlyv1, friendlyv2
```

## Analytical Cost Model of Inference latency on Systolic Array