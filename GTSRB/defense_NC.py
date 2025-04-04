import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    train_path = './data/GTSRB_train_32.npz'
    test_path = './data/GTSRB_test_32.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_train_onehot = np.eye(43)[Y_train]
    Y_test_onehot = np.eye(43)[Y_test]
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot

class TriggerOptimizer:
    def __init__(self, model, target_label, device, img_shape=(3, 32, 32), lr=0.001, lambda_reg=0.01):
        self.model = model
        self.target_label = target_label  # one number
        self.lambda_reg = lambda_reg  # L1 正则化权重
        # 创建触发器 mask 和 pattern
        self.mask = torch.rand(img_shape, requires_grad=True, device=device)
        self.pattern = torch.rand(img_shape, requires_grad=True, device=device)

        # 选择优化器
        self.optimizer = optim.Adam([self.mask, self.pattern], lr=lr)
        self.loss_class = 0
        self.loss_trigger = 0

    def apply_trigger(self, images):
        """ 应用触发器到图像上 """
        mask = self.mask  # 约束 mask 在 (0,1)
        pattern = self.pattern  # 约束 pattern 在 (0,1)
        return (1 - mask) * images + mask * pattern

    def loss_fn(self, images, device):
        """ 计算损失函数，包括分类损失和 L1 正则化 """
        adv_images = self.apply_trigger(images)
        outputs = self.model(adv_images)  # batch*43
        outputs = outputs.to(device)
        outputs = outputs.float()
        self.label_batch = torch.zeros((len(outputs)))
        for i in range(len(outputs)):
            self.label_batch[i] = self.target_label
        self.label_batch = self.label_batch.to(device)
        self.label_batch = self.label_batch.long()
        #print(outputs.device, self.label_batch.device)
        # 计算交叉熵损失，使所有样本都被误分类到 target_label
        classification_loss = F.cross_entropy(outputs, self.label_batch)

        # 计算 L1 规范化，确保触发器尽可能小
        l1_norm = torch.norm(self.mask, p=1)

        # 总损失 = 分类损失 + 触发器稀疏化
        return classification_loss, l1_norm

    def train_trigger(self, train_loader, target_label, device, epochs=100):
        """ 训练触发器 """
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                # self.mask.data += torch.randn_like(self.mask) * 0.01  # 加小扰动
                # self.pattern.data += torch.randn_like(self.pattern) * 0.01
                self.optimizer.zero_grad()
                self.loss_class, self.loss_trigger = self.loss_fn(images, device)
                loss = self.loss_class + self.lambda_reg * self.loss_trigger
                loss.backward()
                self.optimizer.step()
                self.mask.data.clamp_(min=0, max=1)
                self.pattern.data.clamp_(min=0, max=255)
            #if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, loss_c = {self.loss_class:.4f}, loss_t = {self.loss_trigger:.4f}")

            accuracy = self.calculate_accuracy(train_loader, self.model, target_label, device)
            print(f"Accuracy: {accuracy:.4f}")
        trigger = (self.mask * self.pattern).cpu().detach().numpy().transpose(1, 2, 0)
        save_img(trigger, "trigger_"+str(target_label)+".png")

    def return_loss(self):
        return self.loss_trigger

    def calculate_accuracy(self, loader, model, target_label, device):
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                images= self.apply_trigger(images)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                self.label_batch = torch.zeros((len(outputs))).to(device)
                for i in range(len(outputs)):
                    self.label_batch[i] = target_label
                total += self.label_batch.size(0)
                #print(predicted, self.label_batch)
                correct += (predicted == self.label_batch).sum().item()

        accuracy = 100 * correct / total
        return accuracy

def neural_cleanse():
    device = torch.device("cuda")

    X_train, Y_train, _, _, _, _ = load_data()
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = resnet18().to(device)
    model.load_state_dict(torch.load("models/GTSRB_poi_badnets.pt"))  # 加载后门模型
    model.eval()

    loss = np.zeros((43))
    for target_label in range(43):
    #for target_label in [6, 7]:
        print(f"processing label: {target_label}")
        trigger_optimizer = TriggerOptimizer(model, target_label, device)
        trigger_optimizer.train_trigger(train_loader, target_label, device)
        loss[target_label] = trigger_optimizer.return_loss()
    print(loss)
    median = np.median(loss)
    absolute_deviation = np.abs(loss - median)
    mad = np.median(absolute_deviation)
    anomaly_index = absolute_deviation / (mad * 1.4826)
    outliers = anomaly_index > 2
    print(outliers)
    df = pd.DataFrame({
        'Data': loss,
        'Absolute Deviation': absolute_deviation,
        'MAD': mad,
        'Anomaly Index': anomaly_index,
        'Outlier': outliers
    })
    print(df)

neural_cleanse()