"""
Fichier pour la quantization du modèle
""" 

# Imports
import wandb
import torch.nn.functional as F
import torch
import random
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch2trt import torch2trt
assert torch.cuda.is_available(), "Cuda Not Available!"

# Device
device = torch.device("cuda")

# parser 
parser = argparse.ArgumentParser()

# model choice
parser.add_argument("--model", type=str)
args = parser.parse_args()


# wandb config
config = {
    "model": args.model,
    "random_seed": 42,
    "batch_size": 1,
}

run = wandb.init(project=f"test", config=config)
# name wandb run
wandb.run.name = f"{args.model}_quantization"

# Evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()


# Fixer le seed pour la reproductibilité
random.seed(config["random_seed"])
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

# Write Dataloader
data_dir = 'FIRE_DATABASE_3'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir)

# Define the lengths of train, val, and test splits
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply the respective transformations
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']
test_dataset.dataset.transform = data_transforms['test']

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)



# free cache memory 
torch.cuda.empty_cache()

model = torch.load(f"results/{args.model}_pruned_kd.pth").to(device)

# Evalute model
acc, loss = evaluate(model, test_loader)
print('Base Model After Pruning And KD') 
print(f"Accuracy: {acc:.2f}%")
print(f"Loss: {loss:.4f}")

example_input = torch.randn(1, 3, 224, 224).to(device)
model_trt = torch2trt(model,[example_input], fp16_mode=True)

print(model_trt)
# Evalute quantized model
acc, loss = evaluate(model_trt, test_loader)
print('Quantized Model')
print(f"Accuracy: {acc:.2f}%")
print(f"Loss: {loss:.4f}")

base_path = "results"
torch.save(model_trt.state_dict(), f'{base_path}/{args.model}_QUANTIZED.pth')
    
