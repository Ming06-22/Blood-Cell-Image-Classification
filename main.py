import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import *

import torch.multiprocessing as mp 

# apply gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# define loss plotting function
def plot_loss(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.show()
    
# Argument
batch_size = 32
epochs = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load data
train_set = datasets.ImageFolder(root="data/train", transform=transform)
train_set_size = int(len(train_set) * 0.8)
train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, len(train_set) - train_set_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

# initialize model
# model = AdvancedCNN(num_classes=8)
model = MyCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
softmax = nn.Softmax(dim = -1)
training_losses = []

# training
for epoch in range(epochs):
    train_loss = 0
    train_accs = 0
    total = 0
    model.train()
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
     
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        train_accs += (preds == labels).sum().item()
    print(f'Epoch {epoch+1} / {epochs} | train_accs = {train_accs / total} & train_loss = {train_loss / total}')
    training_losses.append((train_loss / total))
    
    model.eval()
    valid_loss = 0.0
    valid_accs = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, preds= torch.max(outputs, 1)
            total += labels.size(0)
            valid_accs += (preds == labels).sum().item()

    print(f'Epoch {epoch+1} / {epochs} | validation_accs = {valid_accs / total} & validation_loss = {valid_loss / total}')

# save model and display loss curve
torch.save(model.state_dict(), './checkpoints/MI_proj1.pth')
plot_loss(training_losses)