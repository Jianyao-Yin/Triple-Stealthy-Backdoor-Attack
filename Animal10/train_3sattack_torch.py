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

# VGG Model
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32

            # Conv Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # Conv Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
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

def extract_triggers(X_train, Y_train, t_class, tri_num, train, trig, f_s_thres, device):  # 1. train model 2. gradcam 3, dct 4, compare and extract
    model_path = "./models"
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 128, 128]
    if train:
        print("start train pre model")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        batch_size = 128
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG11().to(device)
        # print(dir(model))
        summary(model, input_size=(3, 128, 128))
        learning_rate = 0.001
        num_epochs = 20
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("start training")
        train_acc_best = 0
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
            print("{}:{} epoch: {}, loss: {}, train_acc: {:.2f}"
                  .format(mins, secs, epoch, running_loss, train_acc))
            if train_acc > train_acc_best:
                train_acc_best = train_acc
                torch.save(model.state_dict(), os.path.join(model_path, "pre_Animal10.pt"))
        print("training finished")
    else:
        print("skip model training")
        model = VGG11().to(device)
        state_dict_path = os.path.join(model_path, "pre_Animal10.pt")
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    if trig:
        print("start generate trigger")
        target_layer = 'features.17'
        grad_cam = GradCAM(model, target_layer, device)
        triggers = np.zeros((tri_num, 3, 128, 128))
        t_class = torch.tensor(t_class)
        t_class = t_class.to(device)

        n = 0
        i = 0
        while n < tri_num:
            if Y_train[i] == t_class:
                img = X_train[i]
                #save_img(img.transpose(1, 2, 0), '1_trigger_img.png')
                # print(img.shape)
                # save_img(img[0].cpu().numpy().transpose(1, 2, 0)/255, 'img.png')
                # print(np.shape(target_sample))
                img = torch.tensor(np.array(img)).unsqueeze(0)
                img = img.to(device)
                input_tensor = img.clone().detach().float()
                cam = grad_cam.generate_cam(input_tensor, t_class)
                cam = cam.squeeze()  # to array (128, 128)
                #save_img(cam, '2_cam.png')
                #cam = cam / cam.max()
                # print(cam)
                # print(cam.shape)
                img = img.cpu().numpy()
                tailored_img = np.zeros((3, 128, 128))
                for i in range(3):
                    tailored_img[i] = cam * img[0, i]
                # save_img(tailored_img.transpose(1, 2, 0)/255, 'tailored_img.png')
                #save_img(tailored_img.transpose(1, 2, 0)/255, '3_tailored_img.png')
                image_dct = img_dct(img)
                tailored_img_dct = img_dct(tailored_img)
                for i in range(3):
                    for j in range(128):
                        for k in range(128):
                            if (abs((image_dct[i][j][k] - tailored_img_dct[i][j][k]) / image_dct[i][j][k]) < f_s_thres
                                    and image_dct[i][j][k] != 0):
                                triggers[n, i, j, k] = image_dct[i][j][k]
                nonzero_count = np.count_nonzero(triggers[n])
                #save_img(triggers[n].transpose(1, 2, 0)/255, '4_trigger_map.png')
                if nonzero_count > 0:
                    print("trigger num: {} out of {}".format(nonzero_count, triggers[n].size))
                    n += 1
            i += 1
        np.save('3s_trigger.npy', triggers)
    else:
        print("skip trigger generation")
        triggers = np.load('3s_trigger.npy')
        for n in range(len(triggers)):
            nonzero_count = np.count_nonzero(triggers[n])
            print("trigger num: {} out of {}".format(nonzero_count, triggers[n].size))
    # save_img(triggers[0].transpose(1, 2, 0), 'output_image.png')
    return triggers

def gen_poi_sample_3S(x, trigger, p_d_ratio, p_c_r_thres):
    #save_img(x.transpose(1, 2, 0)/255, '5_clean_img.png')
    # trigger embedding
    x_dct = img_dct(x)
    for i in range(3):
        for j in range(128):
            for k in range(128):
                if trigger[0][i][j][k] != 0:
                    x_dct[i][j][k] = x_dct[i][j][k] + p_d_ratio * (trigger[0][i][j][k] - x_dct[i][j][k])
    x_idct = img_idct(x_dct)
    #save_img(x_idct.transpose(1, 2, 0)/255, '6_poisoned_img.png')
    # pixel restriction
    for i in range(3):
        for j in range(128):
            for k in range(128):
                if x_idct[i][j][k] > x[i][j][k] + p_c_r_thres:
                    x_idct[i][j][k] = x[i][j][k] + p_c_r_thres
                elif x_idct[i][j][k] < x[i][j][k] - p_c_r_thres:
                    x_idct[i][j][k] = x[i][j][k] - p_c_r_thres
                if x_idct[i][j][k] > 255:
                    x_idct[i][j][k] = 255
                elif x_idct[i][j][k] < 0:
                    x_idct[i][j][k] = 0
    #save_img(x_idct.transpose(1, 2, 0)/255, '7_f_poisoned_img.png')
    #save_img(np.abs(x-x_idct).transpose(1, 2, 0)/255 * 10, '8_residual_img.png')
    return x_idct.astype(int)

def gen_poi_train_3S(X_train, Y_train, trigger, p_d_ratio, num_poi, p_c_r_thres):
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
            x_p = gen_poi_sample_3S(x_p.transpose(2, 0, 1), trigger, p_d_ratio, p_c_r_thres).transpose(1, 2, 0)
            X_train[n] = x_p  # [n, 128, 128, 3]
            Y_train[n] = target_class
            i += 1
        n += 1

    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    np.savez("./data/train_poisoned_3S.npz", X=X_train, Y=Y_train)
    return X_train, Y_train

def gen_poi_test_3S(X_test, Y_test, trigger, p_d_ratio, p_c_r_thres):
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
            X_p.append(gen_poi_sample_3S(X_test[i].transpose(2, 0, 1), trigger, p_d_ratio, p_c_r_thres).transpose(1, 2, 0))
            Y_p.append(target_class)
    X_p = np.array(X_p)
    Y_p = np.array(Y_p)
    np.savez("./data/test_poisoned_3S.npz", X=X_p, Y=Y_p)
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

def img_dct(img):
    if img.shape[1] == 3:
        img = img.squeeze()
    img_1 = img[0, :, :]
    img_2 = img[1, :, :]
    img_3 = img[2, :, :]
    img_1_dct = dct(dct(img_1.T, type=2, norm='ortho').T, type=2, norm='ortho')
    img_2_dct = dct(dct(img_2.T, type=2, norm='ortho').T, type=2, norm='ortho')
    img_3_dct = dct(dct(img_3.T, type=2, norm='ortho').T, type=2, norm='ortho')
    img_dct = np.zeros((3, 128, 128))
    img_dct[0, :, :] = img_1_dct
    img_dct[1, :, :] = img_2_dct
    img_dct[2, :, :] = img_3_dct
    # img_dct_s = np.abs(np.log2(img_dct) * 0.1)
    # img_show = img_dct_s.transpose(1, 2, 0)
    # save_img(img_show, 'output_image.png')
    return img_dct

def img_idct(img):
    for i in range(3):
        img[i] = idct(idct(img[i].T, norm='ortho').T, norm='ortho')
    return img

def train_3sattack(num_poi=1000, f_s_thres=0.2, p_d_ratio=0.8, p_c_r_thres=20, trig=1):
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 128, 128, 3]
    print(X_train.shape, X_test.shape)

    print("num_poi: {} f_s_thres: {} p_d_ratio: {} p_c_r_thres: {}".format(num_poi, f_s_thres, p_d_ratio, p_c_r_thres))
    triggers = extract_triggers(X_train, Y_train, t_class=5, tri_num=1, train=False, trig=trig, f_s_thres=f_s_thres, device=device)

    print("calculate smr-ssim")
    from skimage.metrics import structural_similarity as ssim
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    smr_ssim_sum = 0
    psnr_sun = 0
    for i in range(10):
        img1 = X_train[i]  # 128*128*3
        img1 = img1.astype(np.uint8).transpose(2, 0, 1)  # 3*128*128
        img2 = gen_poi_sample_3S(img1, triggers, p_d_ratio, p_c_r_thres)
        #print(img1.shape, img2.shape)
        #print(img1.min(), img1.max(), img2.min(), img2.max())
        if img1.shape != img2.shape:
            img2 = Image.fromarray(img2).resize(img1.shape[::-1], Image.LANCZOS)
            img2 = np.array(img2)

        _, ssim_matrix = ssim(img1, img2, full=True, channel_axis=0, data_range=255)
        smr_ssim = (np.sum(np.sqrt(np.clip(ssim_matrix, 0, 1))) / ssim_matrix.size) ** 2
        #print(smr_ssim)
        smr_ssim_sum += smr_ssim
        mse = np.mean((img1 - img2) ** 2)
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        #print(psnr)
        psnr_sun += psnr
    smr_ssim_sum /= 100
    print("average: {}".format(smr_ssim_sum))
    psnr_sun = psnr_sun / 100
    print("psnr: {}".format(psnr_sun))

    print("data process")
    start_time = time.time()
    X_train_p, Y_train_p = gen_poi_train_3S(X_train, Y_train, triggers, p_d_ratio, num_poi, p_c_r_thres)
    X_test_p, Y_test_p = gen_poi_test_3S(X_test, Y_test, triggers, p_d_ratio, p_c_r_thres)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = VGG11().to(device)
    #model = resnet18().to(device)
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
            torch.save(model.state_dict(), os.path.join(model_path, "Animal10_poi_3sattack.pt"))

    print("training finished")

# for num_poi in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]:
#     train_3sattack(num_poi = num_poi)
#
# for f_s_thres in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
#     train_3sattack(f_s_thres = f_s_thres)
#
# for p_c_r_thres in [5, 10, 15, 20, 25, 30]:
#     train_3sattack(p_c_r_thres = p_c_r_thres)

train_3sattack()
