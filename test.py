import sys, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import *
from utils import *

# apply gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# initialize model
batch_size = 32
weight = sys.argv[1]
model = MyCNN().to(device)
model.load_state_dict(torch.load(f"./checkpoints/{weight}", weights_only=False))

# load data
test_set = datasets.ImageFolder(root="data/test", transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# testing
model.eval()
test_accs = total = 0
label_count, pred_count = [], []
with torch.no_grad():
    for batch in tqdm(test_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(imgs)
        _, preds= torch.max(outputs, 1)
        
        total += labels.size(0)
        test_accs += (preds == labels).sum().item()
        label_count.extend(labels.cpu().numpy())
        pred_count.extend(preds.cpu().numpy())

plot_confusion_matrix(label_count, pred_count, test_loader.dataset.classes)
print(f'Test Accuracy : {test_accs / total}')