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
    if not os.path.exists("./res_img/"):
        os.makedirs("./res_img/")
    plt.imshow(img)
    plt.savefig("./res_img/" + name)
    plt.close()
    print("save {}".format(name))

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

def extract_trigger(trigger_image, beta=0.1):
    # 计算触发器图像的 FFT
    fft_trigger = np.fft.fft2(trigger_image, axes=(0, 1))
    fshift_trigger = np.fft.fftshift(fft_trigger, axes=(0, 1))
    # 获取幅值谱 (Amplitude Spectrum)
    amplitude_trigger = np.abs(fshift_trigger)
    print(amplitude_trigger.shape)

    # 生成低频掩码，仅保留低频部分
    H, W, C = trigger_image.shape
    mask = np.zeros((H, W))
    center_H, center_W = H // 2, W // 2
    radius_H, radius_W = int(beta * H), int(beta * W)
    mask[center_H - radius_H:center_H + radius_H, center_W - radius_W:center_W + radius_W] = 1
    # 仅保留低频部分的幅值
    low_freq_amplitude = amplitude_trigger * mask[..., np.newaxis]
    # print(low_freq_amplitude[250, 250])
    return low_freq_amplitude

def gen_poi_sample_FIBA(benign_image, trigger_amplitude, beta=0.1, alpha=0.15):
    # 计算正常样本的 FFT
    #print(benign_image.shape)
    fft_benign = np.fft.fft2(benign_image, axes=(0, 1))
    fshift_benign = np.fft.fftshift(fft_benign, axes=(0, 1))
    # 获取正常样本的幅值和相位
    amplitude_benign = np.abs(fshift_benign)
    phase_benign = np.angle(fshift_benign)

    # 计算新的幅值，仅混合低频区域
    H, W, C = benign_image.shape
    mask = np.zeros((H, W, C))
    center_H, center_W = H // 2, W // 2
    radius_H, radius_W = int(beta * H), int(beta * W)
    mask[center_H - radius_H:center_H + radius_H, center_W - radius_W:center_W + radius_W] = 1
    #print(amplitude_benign.shape, trigger_amplitude.shape)
    new_amplitude = (1 - mask) * amplitude_benign + mask * ((1 - alpha) * amplitude_benign + alpha * trigger_amplitude)

    # 结合相位信息
    fft_poisoned = new_amplitude * np.exp(1j * phase_benign)
    ifftshift_poisoned = np.fft.ifftshift(fft_poisoned, axes=(0, 1))
    # 逆傅立叶变换恢复图像
    poisoned_image = np.fft.ifft2(ifftshift_poisoned, axes=(0, 1)).real
    #print(poisoned_image.max(), poisoned_image.min())
    poisoned_image = np.clip(poisoned_image, 0, 255)
    return poisoned_image

def gen_poi_train_FIBA(X_train, Y_train, trigger_amplitude, num_poi):
    target_class = 7
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)

    n = 0
    i = 0
    while i < num_poi:
        if Y_train[n] != target_class:
            x_p = X_train[n]
            x_p = gen_poi_sample_FIBA(x_p, trigger_amplitude)
            X_train[n] = x_p  # [n, 128, 128, 3]
            Y_train[n] = target_class
            i += 1
        n += 1

    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    #np.savez("./data/train_poisoned_wanet.npz", X=X_train, Y=Y_train)
    return X_train, Y_train

def gen_poi_test_FIBA(X_test, Y_test, trigger_amplitude):
    target_class = 7
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_test)
    np.random.seed(random_seed)
    np.random.shuffle(Y_test)

    X_p = []
    Y_p = []
    for i in range(len(X_test)):
        if Y_test[i] != target_class:
            X_p.append(gen_poi_sample_FIBA(X_test[i], trigger_amplitude))
            Y_p.append(target_class)
    X_p = np.array(X_p)
    Y_p = np.array(Y_p)
    #np.savez("./data/test_poisoned_wanet.npz", X=X_p, Y=Y_p)
    return X_p, Y_p

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

def train_FIBA(num_poi, beta=0.1, alpha=0.8):
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 128, 128, 3]

    print("calculate smr-ssim")
    from skimage.metrics import structural_similarity as ssim
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    trigger = extract_trigger(X_train[random_seed], beta=beta)  # [n, n, 3]
    smr_ssim_sum = 0
    psnr_sun = 0
    for i in range(100):
        img1 = X_train[i]  # 128*128*3
        img1 = img1.astype(np.uint8)  # 128*128*3
        img2 = gen_poi_sample_FIBA(img1, trigger, beta=beta, alpha=alpha)
        print(img1.shape, img2.shape)
        print(img1.min(), img1.max(), img2.min(), img2.max())
        if img1.shape != img2.shape:
            img2 = Image.fromarray(img2).resize(img1.shape[::-1], Image.LANCZOS)
            img2 = np.array(img2)

        _, ssim_matrix = ssim(img1, img2, full=True, channel_axis=2, data_range=255)
        smr_ssim = (np.sum(np.sqrt(np.clip(ssim_matrix, 0, 1))) / ssim_matrix.size) ** 2
        print(smr_ssim)
        smr_ssim_sum += smr_ssim
        mse = np.mean((img1 - img2) ** 2)
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        print(psnr)
        psnr_sun += psnr
    smr_ssim_sum /= 100
    print("average: {}".format(smr_ssim_sum))
    psnr_sun = psnr_sun / 100
    print("psnr: {}".format(psnr_sun))

    print("data process")
    start_time = time.time()
    random_seed = np.random.randint(1000, 9999)
    trigger = extract_trigger(X_train[random_seed], beta=0.1)  # [n, n, 3]
    X_train_p, Y_train_p = gen_poi_train_FIBA(X_train, Y_train, trigger, num_poi)
    X_test_p, Y_test_p = gen_poi_test_FIBA(X_test, Y_test, trigger)
    end_time = time.time() - start_time
    mins, secs = divmod(end_time, 60)
    mins = int(mins)
    secs = int(secs)
    print("data process finished, time usage: {}:{}".format(mins, secs))
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 128, 128]
    X_test = X_test.transpose(0, 3, 1, 2)
    X_train_p = X_train_p.transpose(0, 3, 1, 2)
    X_test_p = X_test_p.transpose(0, 3, 1, 2)
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 输入数据
    # Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)  # 标签 (确保标签是long类型)
    X_train_tensor = torch.tensor(X_train_p, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_p, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    X_test_p_tensor = torch.tensor(X_test_p, dtype=torch.float32)
    Y_test_p_tensor = torch.tensor(Y_test_p, dtype=torch.long)

    # 使用 TensorDataset 将样本和标签组合成一个数据集
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_p_dataset = TensorDataset(X_test_p_tensor, Y_test_p_tensor)

    # 使用 DataLoader 创建可迭代的数据加载器
    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    test_p_loader = DataLoader(dataset=test_p_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model and move it to the appropriate device (GPU if available)
    model_path = "./models"
    #model = VGG11().to(device)
    model = ResNet18().to(device)
    summary(model, input_size=(3, 128, 128))
    learning_rate = 0.001
    num_epochs = 35
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("start training")
    test_acc_best = 0
    start_time = time.time()
    end_time = time.time()
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
        asr = calculate_accuracy(test_p_loader, model, device)
        print("{}:{} epoch: {}, loss: {}, train_acc: {:.2f}, test_acc: {:.2f}, asr: {:.2f}"
              .format(mins, secs, epoch, running_loss, train_acc, test_acc, asr))

        if test_acc > test_acc_best:
            test_acc_best = test_acc
            torch.save(model.state_dict(), os.path.join(model_path, "Animal10_poi_badnets.pt"))
    print("training finished")

train_FIBA(num_poi=1000)