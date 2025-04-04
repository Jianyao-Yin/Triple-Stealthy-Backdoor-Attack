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


# VGG Model
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # Conv Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.classifier(x)
        return x

class resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv2_shortcut = nn.Sequential()

        self.conv3_x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.conv4_x = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv5_x = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        identity = x
        out = self.conv2_x(x)
        out += self.conv2_shortcut(identity)
        out = self.relu(out)

        identity = out
        out = self.conv3_x(out)
        out += self.conv3_shortcut(identity)
        out = self.relu(out)

        identity = out
        out = self.conv4_x(out)
        out += self.conv4_shortcut(identity)
        out = self.relu(out)

        identity = out
        out = self.conv5_x(out)
        out += self.conv5_shortcut(identity)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def load_data():
    train_path = './data/Animal10_train.npz'
    test_path = './data/Animal10_test.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_train_onehot = np.eye(10)[Y_train]
    Y_test_onehot = np.eye(10)[Y_test]
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot

def gen_poi_train():
    return 0

def gen_poi_test():
    return 0

# Function to calculate accuracy
def calculate_accuracy(loader, model, device):
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
# print(np.shape(X_train))
# print(np.shape(Y_train))
# print(np.shape(X_test))
# print(np.shape(Y_test))
# img_dct(X_train)
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
batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model and move it to the appropriate device (GPU if available)
model_path = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18().to(device)
summary(model, input_size=(3, 128, 128))
learning_rate = 0.001
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    train_acc = calculate_accuracy(train_loader, model, device)
    test_acc = calculate_accuracy(test_loader, model, device)
    print("{}:{} epoch: {}, loss: {}, train_acc: {:.2f}, test_acc: {:.2f}"
          .format(mins, secs, epoch, running_loss, train_acc, test_acc))

    if test_acc > test_acc_best:
        test_acc_best = test_acc
        torch.save(model.state_dict(), os.path.join(model_path, "Animal10_cleam.pt"))

print("training finished")