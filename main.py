import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import *
from utils import *

import torch.multiprocessing as mp 

# apply gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")
    
# argument
batch_size = 32
epochs = 15

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
training_losses, validation_losses = [], []
training_accs, validation_accs = [], []

# training
for epoch in range(epochs):
    train_loss = train_acc = total = 0
    model.train()
    print(f'\nEpoch {epoch + 1} / {epochs}')
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
        train_acc += (preds == labels).sum().item()
    print(f'train_acc = {round(train_acc / total, 4)} & train_loss = {round(train_loss / total, 4)}')
    training_accs.append((train_acc / total))
    training_losses.append((train_loss / total))
    
    
    valid_loss = valid_acc = total = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item()
            _, preds= torch.max(outputs, 1)
            total += labels.size(0)
            valid_acc += (preds == labels).sum().item()

    print(f'validation_acc = {round(valid_acc / total, 4)} & validation_loss = {round(valid_loss / total, 4)}')
    validation_accs.append((valid_acc / total))
    validation_losses.append((valid_loss / total))

# save model and save loss curve
torch.save(model.state_dict(), './checkpoints/MI_proj1.pth')
plot_curve(training_accs, validation_accs, "Accuracy")
plot_curve(training_losses, validation_losses, "Loss")