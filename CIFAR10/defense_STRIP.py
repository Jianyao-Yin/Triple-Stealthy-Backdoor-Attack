import os
import time
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

def save_img(img, name):
    if not os.path.exists("./res_img/"):
        os.makedirs("./res_img/")
    plt.imshow(img)
    plt.savefig("./res_img/" + name)
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
        self.linear = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 10),
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
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def load_data():
    test_path = './data/CIFAR10_test.npz'
    test_path_p = './data/test_poisoned_badnets.npz'
    test_data = np.load(test_path, mmap_mode='r')
    test_data_p = np.load(test_path_p, mmap_mode='r')
    X_test = test_data['X']
    Y_test = test_data['Y']
    X_test_p = test_data_p['X']
    Y_test_p = test_data_p['Y']
    return X_test, Y_test, X_test_p, Y_test_p

def STRIP(ratio=0.55):  # ratio:[0, 1], 0: original, 1: replace
    print("load data")
    device = torch.device("cuda")
    X_test, Y_test, X_test_p, Y_test_p = load_data()  # X: [-1, 32, 32, 3]
    model_path = "./models"
    model = resnet18()
    state_dict_path = os.path.join(model_path, "CIFAR10_poi_badnets.pt")
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))
    #device = torch.device("cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_test)
    np.random.seed(random_seed)
    np.random.shuffle(Y_test)
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_test_p)
    np.random.seed(random_seed)
    np.random.shuffle(Y_test_p)

    entropy_field = np.zeros((20,))
    #for i in range(len(X_test_p)):
    for i in range(500):
        test_sample = X_test_p[i]
        test_label = Y_test_p[i]
        label_onehot = torch.tensor(np.expand_dims(np.eye(10)[test_label], axis=0))
        label_onehot = label_onehot.to(device)
        count = 0
        loss_sum = 0
        while count < 100:
            idx = random.randint(0, len(X_test) - 1)
            if test_label == Y_test[idx]:
               pass
            else:
                layer_sample = X_test[idx]
                composed_sample = test_sample * (1-ratio) + layer_sample * ratio
                composed_sample = np.expand_dims(composed_sample, axis=0)
                composed_sample = torch.tensor(composed_sample.transpose(0, 3, 1, 2), dtype=torch.float32)
                composed_sample = composed_sample.to(device)
                output = model(composed_sample)
                loss = F.cross_entropy(output, label_onehot)
                loss_sum += loss.item()
                count += 1
        loss_sum /= 100
        entropy_field[min(int(loss_sum), 19)] += 1
        print(i, loss_sum)
    print(entropy_field)
STRIP()