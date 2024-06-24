"""
Script used to train models 
"""


# Imports
import wandb
import torch.nn.functional as F
import torch
import os
import copy
import random
import numpy as np
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import argparse
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

assert torch.cuda.is_available()


# parser 
parser = argparse.ArgumentParser()

# model choice
parser.add_argument("--model", type=str)
# training parameters 
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")

args = parser.parse_args()


# wandb config
config = {
    "model": args.model,
    "random_seed": 42,
    "batch_size": args.batch_size,
    "optimizer": "SGD",
    "lr": args.lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "CosineLR",
    "loss": "CrossEntropyLoss",
}

run = wandb.init(project=f"FOREST", config=config)
# name wandb run
wandb.run.name = f"{args.model}"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_name, num_classes=3):
    print(f"Loading and modifying model: {model_name}")
    # Get the model class from torchvision.models using the model name
    model_class = getattr(models, model_name.lower())
    
    # Get the corresponding weights class dynamically
    weights_class_name = model_name + '_Weights'
    weights_class = getattr(models, weights_class_name)
    
    # Load the model with the specified weights
    model = model_class(weights=weights_class.DEFAULT)

    # Find the last layer with 1000 output features
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Linear) and module.out_features == 1000:
            # Replace this layer with a new one
            in_features = module.in_features
            new_layer = torch.nn.Linear(in_features, num_classes)
            
            # Set the new layer in the model
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, name, new_layer)
            
            print(f"Modified layer: {name}")
            return model 
        elif isinstance(module, torch.nn.Conv2d) and module.out_channels == 1000:
            # Replace this layer with a new one
            in_channels = module.in_channels
            new_layer = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
            
            # Set the new layer in the model
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, name, new_layer)
            
            print(f"Modified layer: {name}")
            return model 
    else:
        raise ValueError("Could not find a suitable layer to modify")


model = load_model(args.model).to(device)
print(model)

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



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

# Training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    weight_decay=config["weight_decay"],
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=lr, momentum=config["momentum"], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = dict()

  
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward propagation
            loss.backward()

            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        # log les valeurs dans wandb
        wandb.log({"val_acc": acc, "val_loss": val_loss,
                  "lr": optimizer.param_groups[0]["lr"]})

        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)     
    print(f'Best val acc: {best_acc:.2f}')


def main():
    """
    print("Starting main function")
    train(model, train_loader, val_loader, args.epochs, args.lr, weight_decay=config["weight_decay"], save=f"{args.model}.pth")
    print("Training completed")
    acc, loss = evaluate(model, test_loader)
    print(f"Test accuracy: {acc:.2f} | Test loss: {loss:.4f}")
    wandb.log({"test_acc": acc, "test_loss": loss})
    run.finish()
    print("Main function completed")
    """
if __name__ == "__main__":
    main()