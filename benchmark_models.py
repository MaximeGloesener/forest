# iterate over models and calculate size and macs for each model

import torch
from benchmark import get_model_macs, get_model_size, get_num_parameters

# models to benchmark
models = [
    "ConvNeXt_Base",
    "ConvNeXt_Small",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
    "MobileNet_V2",
    "MobileNet_V3_Large",
    "RegNet_Y_16GF",
    "RegNet_Y_32GF",
    "RegNet_Y_8GF",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "SqueezeNet1_1",
    "Swin_V2_B",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
    "ViT_B_16",
    "ViT_B_32"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


# iterate over models
for model_name in models:
    model = torch.load('results/' + model_name + '.pth').to(device)
    example_input = torch.randn(1, 3, 224, 224).to(device)
    macs = get_model_macs(model, example_input)
    size = get_model_size(model)
    num_params = get_num_parameters(model)
    # write into csv file
    with open('results.csv', 'a') as f:
        f.write(f'{model_name},{macs/1e9},{size/MiB},{num_params/1e6}\n')