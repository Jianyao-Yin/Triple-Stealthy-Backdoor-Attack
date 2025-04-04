import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from torch.utils.data import DataLoader, Dataset

def load_data():
    train_path = './data/GTSRB_train.npz' # clean
    train_data = np.load(train_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    test_path = './data/GTSRB_test.npz' # clean
    test_data = np.load(test_path, mmap_mode='r')
    X_test = test_data['X']
    Y_test = test_data['Y']
    test_path_p = './data/test_poisoned_badnets.npz' # clean
    test_data_p = np.load(test_path_p, mmap_mode='r')
    X_test_p = test_data_p['X']
    Y_test_p = test_data_p['Y']
    return X_train, Y_train, X_test, Y_test, X_test_p, Y_test_p
# ============================
# 3. 频率域后门检测器（CNN）
# ============================
class FrequencyClassifier(nn.Module):
    def __init__(self):
        super(FrequencyClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)  # 二分类
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# VGG Model
class VGG11(nn.Module):
    def __init__(self, num_classes=2):
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
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

            # Conv Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.classifier(x)
        return x

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
    def __init__(self):
        super(resnet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear1 = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.linear2 = nn.Sequential(nn.Linear(128, 2))

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
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
# ============================
# 1. 频率域转换（DCT）
# ============================
def dct2(image):
    return cv2.dct(np.float32(image))

def preprocess_image(img):
    for i in range(3):
        img[:, :, i] = dct2(img[:, :, i])
    return img

def gen_poi_sample_FTD(x):
    mask = np.zeros((3, 32, 32))
    trigge_h = np.random.randint(1, 6)
    trigge_w = np.random.randint(1, 6)
    pos_h = np.random.randint(0, 32-trigge_h)
    pos_w = np.random.randint(0, 32-trigge_w)
    mask[:, pos_h:pos_h+trigge_h, pos_w:pos_w+trigge_w] = np.ones((3, trigge_h, trigge_w))
    val = np.random.randint(0, 256)
    #print(mask.shape, x.shape)
    x_p = (1-mask) * x + mask * val
    return x_p
# ============================
# 2. 数据集处理（干净 & 后门样本）
# ============================
class FrequencyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img_np = np.array(img)
        is_poi = np.random.choice([True, False, False, False, False, False, False, False, False, False])
        if is_poi:
            img_np = gen_poi_sample_FTD(img_np)
            label = 1  # poisoned: 1
        else:
            label = 0  # clean: 0
        freq_features = preprocess_image(img_np)
        freq_tensor = torch.tensor(freq_features, dtype=torch.float32)
        return freq_tensor, label

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
# ============================
# 5. 运行实验
# ============================
def FTD(train_clsfr=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, _, X_test, _, X_test_p, _ = load_data()

    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
    Y_train = np.zeros((len(X_train)))
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = 128
    epochs = 500
    lr = 0.001
    clean_data = FrequencyDataset(train_dataset)
    train_loader_p = DataLoader(clean_data, batch_size=batch_size, shuffle=True)

    X_test = X_test.transpose(0, 3, 1, 2)
    for i in range(len(X_test)):
        X_test[i] = preprocess_image(X_test[i])
    Y_test = np.zeros((len(X_test)))
    X_test_p = X_test_p.transpose(0, 3, 1, 2)
    for i in range(len(X_test_p)):
        X_test_p[i] = preprocess_image(X_test_p[i])
    Y_test_p= np.ones((len(X_test_p)))

    X_test_c_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_c_tensor = torch.tensor(Y_test, dtype=torch.long)
    X_test_p_tensor = torch.tensor(X_test_p, dtype=torch.float32)
    Y_test_p_tensor = torch.tensor(Y_test_p, dtype=torch.long)
    test_c_dataset = TensorDataset(X_test_c_tensor, Y_test_c_tensor)
    test_p_dataset = TensorDataset(X_test_p_tensor, Y_test_p_tensor)
    test_c_loader = DataLoader(dataset=test_c_dataset, batch_size=batch_size, shuffle=True)
    test_p_loader = DataLoader(dataset=test_p_dataset, batch_size=batch_size, shuffle=True)

    model = FrequencyClassifier().to(device)
    #model = resnet18().to(device)
    model_path = "./models"
    if train_clsfr:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for data, labels in train_loader_p:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            cle_acc = calculate_accuracy(test_c_loader, model, device)
            poi_acc = calculate_accuracy(test_p_loader, model, device)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader_p):.4f} cle_acc: {cle_acc:.2f}, poi_acc: {poi_acc:.2f}")

        torch.save(model.state_dict(), os.path.join(model_path, "FrequencyClassifier.pt"))
    else:
        state_dict_path = os.path.join(model_path, "FrequencyClassifier.pt")
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    print("Model saved successfully!")
FTD()
