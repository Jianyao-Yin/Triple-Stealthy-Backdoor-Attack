import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision.models import resnet18

import os
import time
from runpy import run_path

import numpy as np
import math
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from GradCAM import GradCAM, GradCAMVisualizer

def save_img(img, name):
    if not os.path.exists("./AC_img/"):
        os.makedirs("./AC_img/")
    plt.imshow(img)
    plt.savefig("./AC_img/" + name)
    plt.close()
    print("save {}".format(name))

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
        self.linear2 = nn.Sequential(nn.Linear(128, 10))

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

def load_data():
    train_path = './data/CIFAR10_train.npz'
    train_data = np.load(train_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    Y_train_onehot = np.eye(10)[Y_train]
    train_path_p = ('./data/train_poisoned_badnets.npz')
    train_data_p = np.load(train_path_p, mmap_mode='r')
    X_train_p = train_data_p['X']
    Y_train_p = train_data_p['Y']
    Y_train_onehot_p = np.eye(10)[Y_train_p]
    return X_train, Y_train, Y_train_onehot, X_train_p, Y_train_p, Y_train_onehot_p

# **4. 提取激活值（最后一层 avgpool 前）**
def get_activations(model, dataloader):
    model.eval()
    activations = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            x = model.conv1(images)
            x = model.bn1(x)
            x = torch.relu(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = nn.functional.avg_pool2d(x, 4)  # 提取 avgpool 层的激活值
            x = x.view(x.size(0), -1)
            x = model.linear1(x)
            x = torch.flatten(x, 1)  # 展平成 1D
            activations.append(x.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    return np.vstack(activations), np.hstack(labels)

# **8. 可视化聚类结果**
def plot_clusters(ica_activations, cluster_labels, label):
    if not os.path.exists("./AC_img/"):
        os.makedirs("./AC_img/")
    plt.figure(figsize=(8, 6))
    plt.scatter(ica_activations[:, 0], ica_activations[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.5)
    plt.title(f"Cluster Visualization - Class {label}")
    plt.xlabel("ICA Component 1")
    plt.ylabel("ICA Component 2")
    plt.colorbar(label="Cluster ID")
    plt.savefig("./AC_img/" + f"AC_{label}")
    plt.close()
    print("save {}".format(label))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **1. 加载 CIFAR-10 数据集**
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
# poisoned_data, poisoned_labels = poison_data(trainset)
# trainloader = torch.utils.data.DataLoader(list(zip(poisoned_data, poisoned_labels)), batch_size=128, shuffle=True)
# model = ResNet18Classifier().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_model(model, trainloader)

X_train, Y_train, Y_train_onehot, X_train_p, Y_train_p, Y_train_onehot_p = load_data()
X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
X_train_p = X_train_p.transpose(0, 3, 1, 2)
sample_list = [[] for _ in range(10)]
count = np.zeros((10))
for i in range(len(X_train_p)):
    for j in range(10):
        if Y_train_p[i] == j:
            sample_list[j].append(X_train_p[i])
            count[j] += 1

model_path = "./models"
model = resnet18().to(device)
state_dict_path = os.path.join(model_path, "CIFAR10_poi_badnets.pt")
model.load_state_dict(torch.load(state_dict_path, weights_only=True))

for i in range(10):
    X_train = sample_list[i]
    X_train = np.array(X_train)
    Y_train = np.zeros((len(X_train)))
    Y_train.fill(i)
    Y_train = np.array(Y_train)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    activations, labels = get_activations(model, train_loader)
    # 先使用 PCA 降维到 10 维
    pca = PCA(n_components=10)
    pca_activations = pca.fit_transform(activations)
    # 再使用 ICA 降维到 2 维
    ica = FastICA(n_components=2)
    ica_activations = ica.fit_transform(pca_activations)
    # K-Means 聚类**
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(ica_activations)  # cluster: [0, 1, 0...]
    # 计算 Silhouette Score 评估聚类效果**
    sil_score = silhouette_score(ica_activations, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    plot_clusters(ica_activations, cluster_labels, i)