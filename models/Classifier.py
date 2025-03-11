import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
import torch.nn as nn
import torch
import torch.nn.functional as F

"""
input dim = 3
output dim = 12
"""


# In project 5, you need to adjust the model architecture.

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(x)
        out = self.bn2(out)

        out = self.relu2(x + out)
        return out


class ResidualConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU()

        self.exconv = nn.Conv2d(
            in_dim, out_dim, kernel_size=1, stride=2, padding=0)
        self.exbn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        exout = self.exconv(x)
        exout = self.exbn(exout)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu2(exout + out)
        return out


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        ################################ you need to modify the cnn model here ################################

        # after convolutoin, the feature map size = ((origin + padding*2_kernel_size) / stride) + 1
        # input_shape=(3,224,224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(
            7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.layer1_1 = ResidualBlock(in_dim=64, out_dim=64)
        self.layer1_2 = ResidualBlock(in_dim=64, out_dim=64)

        self.layer2_1 = ResidualConvBlock(in_dim=64, out_dim=128)
        self.layer2_2 = ResidualBlock(in_dim=128, out_dim=128)

        self.layer3_1 = ResidualConvBlock(in_dim=128, out_dim=256)
        self.layer3_2 = ResidualBlock(in_dim=256, out_dim=256)

        self.layer4_1 = ResidualConvBlock(in_dim=256, out_dim=512)
        self.layer4_2 = ResidualBlock(in_dim=512, out_dim=512)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 8)

        # =================================================================================================== #

    def forward(self, x):

        ################################ you need to modify the cnn model here ################################
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        out = self.layer1_1(out)
        out = self.layer1_2(out)

        out = self.layer2_1(out)
        out = self.layer2_2(out)

        out = self.layer3_1(out)
        out = self.layer3_2(out)

        out = self.layer4_1(out)
        out = self.layer4_2(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        # =================================================================================================== #

        return out