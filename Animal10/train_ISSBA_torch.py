import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchsummary import summary
import threading
import psutil
from datetime import datetime

def monitor_memory():
    """每秒打印一次内存使用情况和当前时间"""
    process = psutil.Process(os.getpid())
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] memory_allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB memory_reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
        time.sleep(1)  # 每秒打印一次
# ============ 1. U-Net 风格的编码器 ============ #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 图像特征提取
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # one-hot信息映射
        self.fc_embed = nn.Linear(100, 128 * 128)  # 将100维信息映射到128x128的特征
        self.conv_embed = nn.Conv2d(1, 512, kernel_size=3, padding=1)  # 用1x128x128形式融合

        # 合并信息
        self.conv_fusion = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        # 还原图像
        self.conv_out1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_out3 = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, img, massage):
        # 处理图像
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 处理信息
        massage_embed = self.fc_embed(massage)  # (batch, 1024)
        massage_embed = massage_embed.view(-1, 1, 128, 128)  # reshape为图像形状
        massage_embed = F.relu(self.conv_embed(massage_embed))  # (batch, 128, 128, 128)

        # 融合信息
        x = torch.cat([x, massage_embed], dim=1)  # (batch, 256, 128, 128)
        x = F.relu(self.conv_fusion(x))  # (batch, 128, 128, 128)

        # 生成隐写图像
        x = F.relu(self.conv_out1(x))
        x = F.relu(self.conv_out2(x))
        stego_img = torch.sigmoid(self.conv_out3(x))  # 输出范围 0-255

        return stego_img

class ImprovedAutoEncoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoEncoder, self).__init__()

        # 编码部分 (下采样)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )  # (16, 64, 64)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # (32, 32, 32)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )  # (64, 16, 16)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )  # (128, 8, 8)

        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )  # (256, 4, 4)

        self.massage_embed = nn.Sequential(
            nn.Linear(100, 64 * 64),  # 将100维信息映射到64x64的特征
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),  # 插入到Sequential
            nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 用1x128x128形式融合
        )

        # 解码部分 (上采样)
        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )  # (128, 8, 8)

        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # 跳跃连接
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )  # (64, 16, 16)

        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),  # 跳跃连接
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # (32, 32, 32)

        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),  # 跳跃连接
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )  # (16, 64, 64)

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16 + 16, 3, kernel_size=3, padding=1),  # 跳跃连接
            nn.Sigmoid()  # 归一化到 [0,1]
        )  # (3, 128, 128)

    def forward(self, x, massage):
        # 编码部分
        e1 = self.encoder1(x)  # (16, 64, 64)
        e2 = self.encoder2(e1)  # (32, 32, 32)
        e3 = self.encoder3(e2)  # (64, 16, 16)
        e4 = self.encoder4(e3)  # (128, 8, 8)
        e5 = self.encoder5(e4)  # (256, 4, 4)

        massage_embed = self.message_embed(massage)
        e5 = torch.cat([e5, massage_embed], dim=1)  # (batch, 512, 128, 128)
        # 解码部分
        d5 = self.decoder5(e5)  # (128, 8, 8)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))  # (64, 16, 16)
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))  # (32, 32, 32)
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  # (16, 64, 64)
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))  # (3, 128, 128)

        return d1
# ============ 2. 轻量级解码器（分类器） ============ #
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )
        # 特征提取
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 预测one-hot编码
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)  # 10位×10分类 = 100维one-hot

    def forward(self, stego_img):
        # 特征提取
        x = self.features(stego_img)
        # 展平
        x = x.view(x.size(0), -1)  # Flatten
        # 全连接层
        x = F.relu(self.fc1(x))
        massage_out = torch.sigmoid(self.fc2(x))  # 限制到[0,1]范围

        return massage_out
# ============ 3. 数据加载 ============ #
class En_De_Dataset(Dataset):
    def __init__(self, x_train):
        self.x_train = x_train
        self.img_size = x_train.shape
        self.length = len(self.x_train)
        self.image = []
        self.label = []

    def one_hot_encode(self, labels):  # [B, 10, 10]
        one_hot_labels = np.zeros((labels.shape[0], 10), dtype=np.float32)
        for i in range(labels.shape[0]):
            one_hot_labels[i, labels[i]] = 1.0  # 在正确位置填 1
        return one_hot_labels.reshape(-1)  # 展平为 100 维

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        rand_idx = np.random.randint(0, self.length)
        img = self.x_train[rand_idx]
        # 生成随机 10 位字符串，每位是 0-9 之间的数字
        self.label = np.random.randint(0, 10, 10)  # (10)
        # 读取图像，转换为 Tensor 并归一化到 [0,1]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # 获取 One-hot 编码的目标信息
        label = torch.tensor(self.one_hot_encode(self.label), dtype=torch.float32).squeeze(0)
        return img, label  # 返回 (3,128,128) [0到1] 的图像和 100 维 One-hot 编码
# ============ 4. 训练过程 ============ #
def train_encoder_decoder(train=False, epochs=2000, lr=0.001):
    X_train, _, _, _, _, _ = load_data()  # X: [-1, 128, 128, 3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImprovedAutoEncoder()
    decoder = Decoder()
    encoder.to(device)
    decoder.to(device)
    # summary(encoder, input_size=[(3, 128, 128), (100,)])
    # summary(decoder, input_size=(3, 128, 128))
    if train:
        encoder.train()
        decoder.train()
        dataset = En_De_Dataset(X_train)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        start_time = time.time()
        best_loss = np.inf
        for epoch in range(epochs):
            total_loss = 0
            for img, massage in dataloader:
                img, massage = img.to(device), massage.to(device)
                poisoned_img = encoder(img, massage)  # 生成中毒样本
                # print(img.max(), img.min(), img.shape, img.type)
                print(poisoned_img.max(), poisoned_img.min())
                decoded_massage = decoder(poisoned_img)  # 解码器尝试恢复类别
                # print(massage.max(), massage.min(), massage.shape, massage.type)
                print(decoded_massage.max(), decoded_massage.min())
                image_loss = torch.sqrt(sum((torch.flatten(img)-torch.flatten(poisoned_img))**2))
                massage_loss = torch.sqrt(sum((torch.flatten(decoded_massage)-torch.flatten(massage))**2))
                print(image_loss, massage_loss)
                loss = image_loss * 0.1 + massage_loss * 10 + (1 - torch.max(decoded_massage)) * 100
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if best_loss > total_loss:
                best_loss = total_loss
                torch.save(encoder, "ISSBA_encoder_full.pth")
                torch.save(decoder, "ISSBA_decoder_full.pth")
            end_time = time.time()
            epoch_time = end_time - start_time
            hors, remains = divmod(epoch_time, 3600)
            mins, secs = divmod(remains, 60)
            hors = int(hors)
            mins = int(mins)
            secs = int(secs)
            print(f"{hors}:{mins}:{secs} Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")
    else:
        encoder = torch.load("ISSBA_encoder_full.pth")
        decoder = torch.load("ISSBA_decoder_full.pth")
    return encoder, decoder

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
class resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet18, self).__init__()

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
        # x = self.maxpool1(x)
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

def gen_poi_sample_ISSBA(x, encoder, massage):  # x: [128, 128, 3], [0, 255]
    #save_img(x.transpose(1, 2, 0), '5_clean_img.png')
    x = x.astype(np.float32) / 255.0  # 归一化到 [0,1]
    x = np.transpose(x, (2, 0, 1))  # 变换通道顺序 (H, W, C) → (C, H, W)
    x = torch.tensor(x).unsqueeze(0)  # 添加 batch 维度 (1, C, H, W)
    massage = torch.tensor(massage)
    x = encoder(x, massage)
    x = x.squeeze(0).cpu().detach().numpy()  # 去掉 batch 维度，转 numpy
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)  # 反归一化 & 转为 uint8
    x = np.transpose(x, (1, 2, 0))  # 调整通道顺序 (C, H, W) → (H, W, C)
    #save_img(x.transpose(1, 2, 0)/255, '6_poisoned_img.png')
    return x.astype(int)

def gen_poi_train_ISSBA(X_train, Y_train, encoder, massage, num_poi):
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
            # print(x_p.shape)
            # plt.imshow(x_p)
            # plt.savefig('clean_image.png')
            x_p = gen_poi_sample_ISSBA(x_p, encoder, massage)
            # print(x_p.shape)
            # plt.imshow(x_p)
            # plt.savefig('p_image.png')
            X_train[n] = x_p  # [n, 128, 128, 3]
            Y_train[n] = target_class
            i += 1
        n += 1

    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    np.savez("./data/train_poisoned_ISSBA.npz", X=X_train, Y=Y_train)
    return X_train, Y_train

def gen_poi_test_ISSBA(X_test, Y_test, encoder, massage):
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
            X_p.append(
                gen_poi_sample_ISSBA(X_test[i], encoder, massage))
            Y_p.append(target_class)
    X_p = np.array(X_p)
    Y_p = np.array(Y_p)
    np.savez("./data/test_poisoned_ISSBA.npz", X=X_p, Y=Y_p)
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

def train_ISSBA(num_poi=1000):
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 128, 128, 3]

    encoder, decoder = train_encoder_decoder()
    encoder.eval()
    decoder.eval()
    encoder.to("cpu")
    decoder.to("cpu")

    massage = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        massage[i, i] = 1  # 生成 one-hot 标签
    massage = massage.flatten()
    massage = np.expand_dims(massage, axis=0)
    print("calculate smr-ssim")
    from skimage.metrics import structural_similarity as ssim
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)

    ssim_sum = 0
    psnr_sum = 0
    for i in range(100):
        img1 = X_train[i]
        img1 = np.array(img1)
        img2 = gen_poi_sample_ISSBA(img1, encoder, massage)
        img2 = np.array(img2)
        if img1.shape != img2.shape:
            img2 = Image.fromarray(img2).resize(img1.shape[::-1], Image.LANCZOS)
            img2 = np.array(img2)
        _, ssim_matrix = ssim(img1, img2, full=True, channel_axis=2, data_range=255)
        # print(ssim_matrix.size)
        ssim_value = (np.sum(np.sqrt(np.clip(ssim_matrix, 0, 1))) / ssim_matrix.size) ** 2
        print(img1.max(), img1.min(), img2.max(), img2.min())
        # save_img(img1, "img1")
        # save_img(img2, "img2")
        ssim_sum += ssim_value
        print(f"SSIM: {ssim_value:.4f}")
        mse = np.mean((img1 - img2) ** 2)
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        if np.isinf(psnr):
            psnr = 25
        psnr_sum += psnr
        print(f"PSNR: {psnr:.4f}")
    ssim_sum /= 100
    print(f"SSIM_average: {ssim_sum:.4f}")
    psnr_sum /= 100
    print(f"PSNR_average: {psnr_sum:.4f}")

    print("data process")
    start_time = time.time()

    X_train_p, Y_train_p = gen_poi_train_ISSBA(X_train, Y_train, encoder, massage, num_poi)
    X_test_p, Y_test_p = gen_poi_test_ISSBA(X_test, Y_test, encoder, massage)
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
    # model = VGG11().to(device)
    model = resnet18().to(device)
    summary(model, input_size=(3, 128, 128))
    learning_rate = 0.001
    num_epochs = 50
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
            torch.save(model.state_dict(), os.path.join(model_path, "Animal10_poi_ISSBA.pt"))

    print("training finished")

# monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
# monitor_thread.start()
train_ISSBA()