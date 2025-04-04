import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchsummary import summary
from GradCAM import GradCAM, GradCAMVisualizer

def save_img(img, name):
    if not os.path.exists("./cam_img/"):
        os.makedirs("./cam_img/")
    plt.imshow(img)
    plt.savefig("./cam_img/" + name)
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
        self.linear2 = nn.Sequential(nn.Linear(128, 43))

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
    test_path = './data/GTSRB_test_32.npz'
    test_data = np.load(test_path, mmap_mode='r')
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_test_onehot = np.eye(43)[Y_test]
    test_path_p = './data/test_poisoned_3S.npz'
    test_data_p = np.load(test_path_p, mmap_mode='r')
    X_test_p = test_data_p['X']
    Y_test_p = test_data_p['Y']
    Y_test_onehot_p = np.eye(43)[Y_test_p]
    return X_test, Y_test, Y_test_onehot, X_test_p, Y_test_p, Y_test_onehot_p

def gradcam():
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test, Y_test, _, X_test_p, Y_test_p, _ = load_data()  # X: [-1, 32, 32, 3]
    X_test = X_test.transpose((0, 3, 1, 2))  # [-1, 3, 32, 32]
    X_test_p = X_test_p.transpose((0, 3, 1, 2))
    model_path = "./models"
    model = resnet18().to(device)
    state_dict_path = os.path.join(model_path, "GTSRB_poi_3Sattack.pt")
    model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location=torch.device('cpu')))
    summary(model, input_size=(3, 32, 32))

    target_layer = 'layer4'
    grad_cam = GradCAM(model, target_layer, device)
    cmap = cm.get_cmap('hot')
    for i in range(100):
        x = torch.tensor(X_test_p[i], dtype=torch.float32).to(device)
        x = x.unsqueeze(0)
        output = model(x)
        _, predicted = torch.max(output, 1)
        cam = grad_cam.generate_cam(x, predicted)
        cam = cam.squeeze()  # to array (32, 32)
        cam = cam / cam.max()
        cam_rgb = cmap(cam)[:, :, :3]
        x = x[0].cpu().detach().numpy().transpose((1, 2, 0))
        #print(x.shape, x.dtype, cam_rgb.shape, cam_rgb.dtype)
        x_show = (x/255 + cam_rgb) / 2
        save_img(x_show, f"cam_{i}.png")

gradcam()