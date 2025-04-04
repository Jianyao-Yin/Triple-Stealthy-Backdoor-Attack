import os
import time
import numpy as np
import math
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

# Models
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class resnet18(nn.Module):
    def __init__(self, num_classes=100):
        super(resnet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=100):
        super(DenseNet121, self).__init__()
        num_blocks = [6, 12, 24, 16]
        num_channels = 2 * growth_rate

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第一个 Dense Block
        layers = []
        for _ in range(num_blocks[0]):
            layers.append(self._make_bottleneck_layer(num_channels, growth_rate))
            num_channels += growth_rate
        self.dense_block1 = nn.Sequential(*layers)
        self.transition1 = self._make_transition_layer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # 第二个 Dense Block
        layers = []
        for _ in range(num_blocks[1]):
            layers.append(self._make_bottleneck_layer(num_channels, growth_rate))
            num_channels += growth_rate
        self.dense_block2 = nn.Sequential(*layers)
        self.transition2 = self._make_transition_layer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # 第三个 Dense Block
        layers = []
        for _ in range(num_blocks[2]):
            layers.append(self._make_bottleneck_layer(num_channels, growth_rate))
            num_channels += growth_rate
        self.dense_block3 = nn.Sequential(*layers)
        self.transition3 = self._make_transition_layer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # 第四个 Dense Block
        layers = []
        for _ in range(num_blocks[3]):
            layers.append(self._make_bottleneck_layer(num_channels, growth_rate))
            num_channels += growth_rate
        self.dense_block4 = nn.Sequential(*layers)

        # 最终的 BatchNorm, 全局平均池化和全连接层
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))  # 修改为输出 2x2 的特征图
        self.fc = nn.Linear(num_channels * 2 * 2, num_classes)

    def _make_bottleneck_layer(self, in_channels, growth_rate):
        layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layers

    def _make_transition_layer(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        return layers

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)

        out = self.dense_block1(out)
        out = self.transition1(out)

        out = self.dense_block2(out)
        out = self.transition2(out)

        out = self.dense_block3(out)
        out = self.transition3(out)

        out = self.dense_block4(out)

        out = self.relu(self.bn2(out))
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def load_data():
    train_path = './data/CIFAR100_train.npz'
    test_path = './data/CIFAR100_test.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_train_onehot = np.eye(100)[Y_train]
    Y_test_onehot = np.eye(100)[Y_test]
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot

def gen_poi_train():
    return 0

def gen_poi_test():
    return 0

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def img_dct(X):
    for i in range(100):
        img = X_train[i]
        img_1 = img[:, :, 0]
        img_2 = img[:, :, 1]
        img_3 = img[:, :, 2]
        img_1_dct = dct(dct(img_1.T, type=2, norm='ortho').T, type=2, norm='ortho')
        img_2_dct = dct(dct(img_2.T, type=2, norm='ortho').T, type=2, norm='ortho')
        img_3_dct = dct(dct(img_3.T, type=2, norm='ortho').T, type=2, norm='ortho')
        img_dct = np.zeros((32, 32, 3))
        img_dct[:, :, 0] = img_1_dct
        img_dct[:, :, 1] = img_2_dct
        img_dct[:, :, 2] = img_3_dct
        for j in range(32):
            for k in range(32):
                for l in range(3):
                    img_dct[j, k, l] = np.abs(img_dct[j, k, l] * 0.01)
        plt.imshow(img_dct)
        plt.show()
        print(i)

print("load data")
X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()
# img_dct(X_train)
# print(np.shape(X_train))
# print(np.shape(Y_train))
# print(np.shape(X_test))
# print(np.shape(Y_test))
X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 输入数据
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)  # 标签 (确保标签是long类型)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# 使用 TensorDataset 将样本和标签组合成一个数据集
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# 使用 DataLoader 创建可迭代的数据加载器
batch_size = 1024
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model and move it to the appropriate device (GPU if available)
model_path = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = DenseNet121().to(device)
model = resnet18().to(device)
summary(model, input_size=(3, 32, 32))
learning_rate = 0.001
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

test_acc_best = 0
start_time = time.time()
end_time = time.time()

print("start training")
start_time = time.time()
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** max(0, (epoch - 10)))
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f'Current Learning Rate: {current_lr:.6f}')
    end_time = time.time()
    epoch_time = end_time - start_time
    mins, secs = divmod(epoch_time, 60)
    mins = int(mins)
    secs = int(secs)
    train_acc = calculate_accuracy(train_loader, model)
    test_acc = calculate_accuracy(test_loader, model)
    print("{}:{} epoch: {}, loss: {}, train_acc: {}, test_acc: {}"
          .format(mins, secs, epoch, running_loss, train_acc, test_acc))

    if test_acc > test_acc_best:
        test_acc_best = test_acc
        torch.save(model.state_dict(), os.path.join(model_path, "CIFAR10_model_clean.pt"))

print("training finished")