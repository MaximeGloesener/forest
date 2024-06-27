# iterate over models and calculate size and macs for each model

import torch
from benchmark import measure_latency_gpu

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


# iterate over models
for model_name in models:
    model = torch.load('results/' + model_name + '.pth').to(device)
    example_input = torch.randn(1, 3, 224, 224).to(device)
    mean_syn_gpu, std_syn_gpu, fps = measure_latency_gpu(model, example_input) 
    # write into csv file
    with open('results.csv', 'a') as f:
        f.write(f'{model_name}, {fps}\n')

# same for pruned models
for model_name in models:
    model = torch.load('results/' + model_name + '_pruned_kd.pth').to(device)
    example_input = torch.randn(1, 3, 224, 224).to(device)
    mean_syn_gpu, std_syn_gpu, fps = measure_latency_gpu(model, example_input) 
    # write into csv file
    with open('results.csv', 'a') as f:
        f.write(f'{model_name}+"_pruned_kd.pth", {fps}\n')
