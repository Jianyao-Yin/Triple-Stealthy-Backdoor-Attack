import os
import time
from runpy import run_path

import numpy as np
import math
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, num_classes=100):
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
            #nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

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

class BasicBlock_W(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, drop_rate=0.0):
        super(BasicBlock_W, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=100, widen_factor=10, drop_rate=0.3):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6
        block = BasicBlock_W

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, n_channels[0], n_channels[1], n, 1, drop_rate)
        self.layer2 = self._make_layer(block, n_channels[1], n_channels[2], n, 2, drop_rate)
        self.layer3 = self._make_layer(block, n_channels[2], n_channels[3], n, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.linear = nn.Linear(n_channels[3], num_classes)

    def _make_layer(self, block, in_planes, out_planes, n_blocks, stride, drop_rate):
        layers = []
        for i in range(n_blocks):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def load_data():
    train_path = './data/CIFAR100_train.npz'
    test_path = './data/CIFAR100_test.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']
    Y_train_onehot = np.eye(100)[Y_train]
    Y_test_onehot = np.eye(100)[Y_test]
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot

def extract_triggers(X_train, Y_train, t_class, tri_num, train, trig, f_s_thres, device):  # 1. train model 2. gradcam 3, dct 4, compare and extract
    model_path = "./models"
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 32, 32]
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
        summary(model, input_size=(3, 32, 32))
        learning_rate = 0.0003
        num_epochs = 50
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
            train_acc = calculate_accuracy(train_loader, model)
            print("{}:{} epoch: {}, loss: {}, train_acc: {:.2f}"
                  .format(mins, secs, epoch, running_loss, train_acc))
            if train_acc > train_acc_best:
                train_acc_best = train_acc
                torch.save(model.state_dict(), os.path.join(model_path, "pre_CIFAR100.pt"))
        print("training finished")
    else:
        print("skip model training")
        model = VGG11().to(device)
        state_dict_path = os.path.join(model_path, "pre_CIFAR100.pt")
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    if trig:
        print("start generate trigger")
        target_sample = []
        n = 0
        i = 0
        while n < tri_num:
            if Y_train[i] == t_class:
                target_sample.append(X_train[i])
                # save_img(X_train[i].transpose(1, 2, 0), '1_trigger_img.png')
                n += 1
            i += 1
        target_sample = torch.tensor(np.array(target_sample))
        target_sample = target_sample.to(device)
        t_class = torch.tensor(t_class)
        t_class = t_class.to(device)
        #print(dir(model))
        target_layer = 'features.17'
        grad_cam = GradCAM(model, target_layer, device)
        triggers = np.zeros((tri_num, 3, 32, 32))
        for n in range(tri_num):
            img = target_sample[n].unsqueeze(0)
            print(img[0].shape)
            # print(np.shape(target_sample))
            input_tensor = img.clone().detach().float()
            cam = grad_cam.generate_cam(input_tensor, t_class)
            cam = cam.squeeze()  # to array (32, 32)
            cam = cam / cam.max()
            # save_img(cam, '2_cam.png')
            # print(cam.shape)
            img = img.cpu().numpy()
            tailored_img = np.zeros((3, 32, 32))
            for i in range(3):
                tailored_img[i] = cam * img[0, i]
            # save_img(tailored_img.transpose(1, 2, 0) / 255, '3_tailored_img.png')
            image_dct = img_dct(img)
            tailored_img_dct = img_dct(tailored_img)
            for i in range(3):
                for j in range(32):
                    for k in range(32):
                        if (abs((image_dct[i][j][k] - tailored_img_dct[i][j][k]) / image_dct[i][j][k]) < f_s_thres
                            and image_dct[i][j][k] != 0):
                            triggers[n, i, j, k] = image_dct[i][j][k]
            nonzero_count = np.count_nonzero(triggers[n])
            print("trigger num: {} out of {}".format(nonzero_count, triggers[n].size))
        np.save('3s_trigger.npy', triggers)
    else:
        print("skip trigger generation")
        triggers = np.load('3s_trigger.npy')
        for n in range(len(triggers)):
            nonzero_count = np.count_nonzero(triggers[n])
            print("trigger num: {} out of {}".format(nonzero_count, triggers[n].size))
    # save_img(triggers[0].transpose(1, 2, 0), '4_trigger_map.png')
    return triggers

def gen_poi_sample_3S(x, trigger, rp, p_c_r_thres):
    # trigger embedding
    # save_img(x.transpose(1, 2, 0), '5_clean_img.png')
    x_dct = img_dct(x)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if trigger[0][i][j][k] != 0:
                    x_dct[i][j][k] = x_dct[i][j][k] + rp * (trigger[0][i][j][k] - x_dct[i][j][k])
    x_idct = img_idct(x_dct)
    # save_img(x_idct.transpose(1, 2, 0) / 255, '6_poisoned_img.png')
    # pixel restriction
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if x_idct[i][j][k] > x[i][j][k] + p_c_r_thres:
                    x_idct[i][j][k] = x[i][j][k] + p_c_r_thres
                elif x_idct[i][j][k] < x[i][j][k] - p_c_r_thres:
                    x_idct[i][j][k] = x[i][j][k] - p_c_r_thres
                if x_idct[i][j][k] > 255:
                    x_idct[i][j][k] = 255
                elif x_idct[i][j][k] < 0:
                    x_idct[i][j][k] = 0
    # save_img(x_idct.transpose(1, 2, 0) / 255, '7_f_poisoned_img.png')
    return x_idct.astype(int)

def gen_poi_train_3S(X_train, Y_train, trigger, rp, num_poi, p_c_r_thres):
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
            x_p = gen_poi_sample_3S(x_p.transpose(2, 0, 1), trigger, rp, p_c_r_thres).transpose(1, 2, 0)
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
    np.savez("./data/train_poisoned_3S.npz", X=X_train, Y=Y_train)
    return X_train, Y_train

def gen_poi_test_3S(X_test, Y_test, trigger, rp, p_c_r_thres):
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
            X_p.append(gen_poi_sample_3S(X_test[i].transpose(2, 0, 1), trigger, rp, p_c_r_thres).transpose(1, 2, 0))
            Y_p.append(target_class)
    X_p = np.array(X_p)
    Y_p = np.array(Y_p)
    np.savez("./data/test_poisoned_3S.npz", X=X_p, Y=Y_p)
    return X_p, Y_p

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    img_dct = np.zeros((3, 32, 32))
    img_dct[0, :, :] = img_1_dct
    img_dct[1, :, :] = img_2_dct
    img_dct[2, :, :] = img_3_dct
    # img_dct_s = np.abs(np.log2(img_dct) * 0.1)
    # img_show = img_dct_s.transpose(1, 2, 0)
    # plt.imshow(img_show)
    # plt.savefig('output_image.png')
    return img_dct

def img_idct(img):
    for i in range(3):
        img[i] = idct(idct(img[i].T, norm='ortho').T, norm='ortho')
    return img

def train_3sattack(num_poi=2000, f_s_thres=0.05, p_d_ratio=0.56, p_c_r_thres=20, trig=1):
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 32, 32, 3]

    print("num_poi: {} f_s_thres: {} p_d_ratio: {} p_c_r_thres: {}, trig: {}".format(num_poi, f_s_thres, p_d_ratio, p_c_r_thres, trig))
    triggers = extract_triggers(X_train, Y_train, t_class=5, tri_num=1, train=False, trig=trig, f_s_thres=f_s_thres, device=device)

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
        img2 = gen_poi_sample_3S(img1.transpose(2, 0, 1), triggers, p_d_ratio, p_c_r_thres)
        img2 = np.array(img2).transpose(1, 2, 0)
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

    X_train_p, Y_train_p = gen_poi_train_3S(X_train, Y_train, triggers, p_d_ratio, num_poi, p_c_r_thres)
    X_test_p, Y_test_p = gen_poi_test_3S(X_test, Y_test, triggers, p_d_ratio, p_c_r_thres)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = VGG11().to(device)
    #model = resnet18().to(device)
    model = WideResNet().to(device)
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
        train_acc = calculate_accuracy(train_loader, model)
        test_acc = calculate_accuracy(test_loader, model)
        asr = calculate_accuracy(test_p_loader, model)
        print("{}:{} epoch: {}, loss: {}, train_acc: {:.2f}, test_acc: {:.2f}, asr: {:.2f}"
              .format(mins, secs, epoch, running_loss, train_acc, test_acc, asr))

        if test_acc > test_acc_best:
            test_acc_best = test_acc
            torch.save(model.state_dict(), os.path.join(model_path, "CIFAR100_poi_badnets.pt"))
    print("training finished")

num_poi_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
f_s_thres_list = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40,
                  0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80]
p_d_ratio_list = [1, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60,
                  0.56, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20]
p_c_r_thres_list = [5, 10, 15, 20, 25, 30, 35, 40]

# for round in range(len(num_poi_list)):
#     train_3sattack(num_poi=num_poi_list[round], trig=1)
# for round in range(len(f_s_thres_list)):
#     train_3sattack(f_s_thres=f_s_thres_list[round], trig=1)
# for round in range(len(p_d_ratio_list)):
#     train_3sattack(p_d_ratio=p_d_ratio_list[round], trig=1)
# for round in range(len(p_c_r_thres_list)):
#     train_3sattack(p_c_r_thres=p_c_r_thres_list[round], trig=1)
train_3sattack()
