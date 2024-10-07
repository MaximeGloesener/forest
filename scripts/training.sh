python compression.py --model RegNet_Y_128GF --batch-size 8
python compression.py --model RegNet_Y_32GF
python compression.py --model RegNet_Y_16GF

python compression.py --model EfficientNet_V2_L
python compression.py --model EfficientNet_V2_M
python compression.py --model EfficientNet_V2_S


python compression.py --model ConvNeXt_Base
python compression.py --model ConvNeXt_Small
python compression.py --model ConvNeXt_Large --batch-size 8



python compression.py --model VGG11
python compression.py --model VGG13
python compression.py --model VGG16
python compression.py --model VGG19


python compression.py --model ResNet50
python compression.py --model ResNet101
python compression.py --model ResNet152
python compression.py --model ResNet34
python compression.py --model ResNet18


python quantization.py --model RegNet_Y_128GF --batch-size 8
python quantization.py --model RegNet_Y_32GF
python quantization.py --model RegNet_Y_16GF

python quantization.py --model EfficientNet_V2_L
python quantization.py --model EfficientNet_V2_M
python quantization.py --model EfficientNet_V2_S


python quantization.py --model ConvNeXt_Base
python quantization.py --model ConvNeXt_Small
python quantization.py --model ConvNeXt_Large --batch-size 8



python quantization.py --model VGG11
python quantization.py --model VGG13
python quantization.py --model VGG16
python quantization.py --model VGG19


python quantization.py --model ResNet50
python quantization.py --model ResNet101
python quantization.py --model ResNet152
python quantization.py --model ResNet34
python quantization.py --model ResNet18