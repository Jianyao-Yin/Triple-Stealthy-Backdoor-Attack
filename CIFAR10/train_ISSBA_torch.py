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

# ============ 1. U-Net 风格的编码器 ============ #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 图像特征提取
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # one-hot信息映射
        self.fc_embed = nn.Linear(100, 32 * 32)  # 将100维信息映射到32x32的特征
        self.conv_embed = nn.Conv2d(1, 128, kernel_size=3, padding=1)  # 用1x32x32形式融合

        # 合并信息
        self.conv_fusion = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # 还原图像
        self.conv_out1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_out3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, img, message):
        # 处理图像
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 处理信息
        message_embed = self.fc_embed(message)  # (batch, 1024)
        message_embed = message_embed.view(-1, 1, 32, 32)  # reshape为图像形状
        message_embed = F.relu(self.conv_embed(message_embed))  # (batch, 128, 32, 32)

        # 融合信息
        x = torch.cat([x, message_embed], dim=1)  # (batch, 256, 32, 32)
        x = F.relu(self.conv_fusion(x))  # (batch, 128, 32, 32)

        # 生成隐写图像
        x = F.relu(self.conv_out1(x))
        x = F.relu(self.conv_out2(x))
        stego_img = torch.sigmoid(self.conv_out3(x))  # 输出范围 0-255

        return stego_img
# ============ 2. 轻量级解码器（分类器） ============ #
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 特征提取
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 预测one-hot编码
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 100)  # 10位×10分类 = 100维one-hot

    def forward(self, stego_img):
        # 特征提取
        x = F.relu(self.conv1(stego_img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.view(x.size(0), -1)  # Flatten

        # 全连接层
        x = F.relu(self.fc1(x))
        message_out = torch.sigmoid(self.fc2(x))  # 限制到[0,1]范围

        return message_out
# ============ 3. 数据加载 ============ #
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(32, 32, 3)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.image = []
        self.label = []

    def one_hot_encode(self, labels):  # [B, 10, 10]
        one_hot_labels = np.zeros((labels.shape[0], 10), dtype=np.float32)
        for i in range(labels.shape[0]):
            one_hot_labels[i, labels[i]] = 1.0  # 在正确位置填 1
        return one_hot_labels.reshape(-1)  # 展平为 100 维

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像 (num_samples, 3, 32, 32)，范围 0-255
        self.image = np.random.randint(0, 256, self.img_size, dtype=np.uint8)
        # 生成随机 10 位字符串，每位是 0-9 之间的数字
        self.label = np.random.randint(0, 10, 10)  # (10)
        # 读取图像，转换为 Tensor 并归一化到 [0,1]
        img = torch.tensor(self.image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # 获取 One-hot 编码的目标信息
        label = torch.tensor(self.one_hot_encode(self.label), dtype=torch.float32).squeeze(0)
        return img, label  # 返回 (3,32,32) [0到1] 的图像和 100 维 One-hot 编码
# ============ 4. 训练过程 ============ #
def train_encoder_decoder(train=True, epochs=20, lr=0.001):
    encoder = Encoder()
    decoder = Decoder()
    if train:
        encoder.train()
        decoder.train()
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        image_loss = nn.L1Loss()  # 图像重构损失
        massage_loss = nn.BCELoss()  # 信息隐写损失
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        start_time = time.time()
        best_loss = np.inf
        for epoch in range(epochs):
            total_loss = 0
            for img, message in dataloader:
                poisoned_img = encoder(img, message)  # 生成中毒样本
                # print(img.max(), img.min(), img.shape, img.type)
                print(poisoned_img.max(), poisoned_img.min())
                decoded_message = decoder(poisoned_img)  # 解码器尝试恢复类别
                # print(message.max(), message.min(), message.shape, message.type)
                print(decoded_message.max(), decoded_message.min())
                image_loss = torch.sqrt(sum((torch.flatten(img)-torch.flatten(poisoned_img))**2))
                massage_loss = torch.sqrt(sum((torch.flatten(decoded_message)-torch.flatten(message))**2))
                print(image_loss, massage_loss)
                loss = image_loss + massage_loss
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
    train_path = './data/CIFAR10_train.npz'
    test_path = './data/CIFAR10_test.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_train_onehot = np.eye(10)[Y_train]
    Y_test_onehot = np.eye(10)[Y_test]
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot

def gen_poi_sample_ISSBA(x, encoder, massage):  # x: [32, 32, 3], [0, 255]
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
            X_train[n] = x_p  # [n, 32, 32, 3]
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
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 32, 32, 3]

    encoder, decoder = train_encoder_decoder(train=False, epochs=2000, lr=0.001)
    encoder.eval()
    decoder.eval()
    massage = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        massage[i, i] = 1  # 生成 one-hot 标签
    massage = massage.flatten()
    #print(massage.shape)
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
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
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
    summary(model, input_size=(3, 32, 32))
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
            torch.save(model.state_dict(), os.path.join(model_path, "CIFAR10_poi_ISSBA.pt"))

    print("training finished")

train_ISSBA()