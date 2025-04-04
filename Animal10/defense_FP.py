import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision.models import resnet18
from torchsummary import summary
import numpy as np
import pandas as pd

# 定义残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义输入为 128x128x3 的十分类 ResNet-18 模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 残差块 1
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 残差块 2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # 残差块 3
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # 残差块 4
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        #x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def load_data():
    test_path = './data/Animal10_test.npz' # clean
    test_data = np.load(test_path, mmap_mode='r')
    X_test = test_data['X']
    Y_test = test_data['Y']
    test_path_p = './data/test_poisoned_3S.npz' # poisoned
    test_data_p = np.load(test_path_p, mmap_mode='r')
    X_test_p = test_data_p['X']
    Y_test_p = test_data_p['Y']
    return X_test, Y_test, X_test_p, Y_test_p

def hook_fn(module, input, output, layer_name):
    """ 记录神经元激活值，区分 Conv2d 和 Linear """
    if layer_name not in activations:
        activations[layer_name] = []

    output = output.detach().cpu()

    if isinstance(module, nn.Conv2d):
        # 对 Conv2d，取通道维度的均值
        activations[layer_name].append(output.mean(dim=[0, 2, 3]))  # (channels,)
    elif isinstance(module, nn.Linear):
        # 对 Linear，只取特征维度的均值
        activations[layer_name].append(output.mean(dim=0))  # (features,)
    else:
        print(f"Skipping {layer_name}: not Conv2d or Linear")
# 运行测试集，获取激活度
def get_neuron_activations(model, dataloader):
    """ 计算所有层神经元的平均激活度 """
    model.eval()
    global activations
    activations = {}  # 清空激活记录

    # 注册钩子
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):  # 只对 Conv2d 和 Linear 层注册
            hooks.append(layer.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name)))

    # 遍历数据集，计算激活值
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)  # 触发前向传播

    # 计算平均激活值
    for key in activations:
        activations[key] = torch.stack(activations[key]).mean(dim=0)  # 计算全数据集的平均激活值

    # 取消钩子
    for hook in hooks:
        hook.remove()

    return activations
# 6. 剪枝（基于神经元激活度）
def prune_neurons(model, activations, device, prune_ratio=0.3):
    """ 剪去在干净数据上激活值最低的神经元 """
    #print(activations.keys())
    count = 0
    eff = 0
    for name, layer in model.named_modules():
        count += 1
        # if isinstance(layer, nn.ReLU) and name in activations:
        if name in activations.keys():
            eff += 1
            #print(count, eff, name)
            activation_values = activations[name]
            threshold = torch.quantile(activation_values, prune_ratio)
            mask = (activation_values > threshold).float()
            expand_shape = (slice(None),) + (np.newaxis,) * (layer.weight.data.ndim - 1)
            mask = torch.tensor(mask[expand_shape], dtype=torch.bool).to(device)
            #print(mask.shape, layer.weight.data.shape)
            layer.weight.data *= mask
    return model
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

X_test, Y_test, X_test_p, Y_test_p = load_data()  # X: [-1, 32, 32, 3]
# to munpy, transpose, to tensor, to cuda,
X_test = X_test.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
X_test_p = X_test_p.transpose(0, 3, 1, 2)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
X_test_p_tensor = torch.tensor(X_test_p, dtype=torch.float32)
Y_test_p_tensor = torch.tensor(Y_test_p, dtype=torch.long)

# 使用 TensorDataset 将样本和标签组合成一个数据集
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_p_dataset = TensorDataset(X_test_p_tensor, Y_test_p_tensor)

# 使用 DataLoader 创建可迭代的数据加载器
batch_size = 128
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
test_p_loader = DataLoader(dataset=test_p_dataset, batch_size=batch_size, shuffle=True)

# 2. 定义 ResNet-18 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models"
model = ResNet18().to(device)
state_dict_path = os.path.join(model_path, "Animal10_poi_3sattack.pt")
model.load_state_dict(torch.load(state_dict_path, weights_only=True))
#summary(model, input_size=(3, 32, 32))

activations = {}
hooks = []
# 计算神经元的激活度
print("Computing neuron activations...")
neuron_activations = get_neuron_activations(model, test_loader)
print(neuron_activations.keys())
neuron_activations.popitem()
print(neuron_activations.keys())
data = np.zeros((3, 100))
count = 0
for i in range(4, 202, 4):
    prune_ratio = i / 2000
    print(f"Applying Fine-Pruning: {prune_ratio*100}%")
    model_pruned = prune_neurons(model, neuron_activations, device, prune_ratio=prune_ratio)
    clean_acc = calculate_accuracy(test_loader, model_pruned, device)
    poi_acc = calculate_accuracy(test_p_loader, model_pruned, device)
    print(f"Fine-Pruning complete. clean_acc: {clean_acc}, poi_acc: {poi_acc}")
    data[0, count] = prune_ratio * 100
    data[1, count] = clean_acc
    data[2, count] = poi_acc
    count += 1
data = data.T
df = pd.DataFrame(data)
df.to_excel("output_FP.xlsx", index=False, header=False)