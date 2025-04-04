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

class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # Output: (batch_size, 16, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 16, 14, 14)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Output: (batch_size, 32, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, 7, 7)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # Output: (batch_size, 64, 7, 7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 64, 3, 3)
        )

        self.fc = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_data():
    train_path = './data/MNIST_train.npz'
    test_path = './data/MNIST_test.npz'
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
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 28, 28]
    if train:
        print("start train pre model")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        batch_size = 128
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNIST_CNN().to(device)
        # print(dir(model))
        summary(model, input_size=(1, 28, 28))
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
                torch.save(model.state_dict(), os.path.join(model_path, "pre_MNIST.pt"))
        print("training finished")
    else:
        print("skip model training")
        model = MNIST_CNN().to(device)
        state_dict_path = os.path.join(model_path, "pre_MNIST.pt")
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    X_train = X_train.transpose(0, 2, 3, 1)
    if trig:
        print("start generate trigger")
        target_sample = []
        n = 0
        i = 0
        while n < tri_num:
            if Y_train[i] == t_class:
                print(X_train[i].shape)
                target_sample.append(X_train[i])
                save_img(X_train[i], '1_trigger_img.png')
                n += 1
            i += 1
        target_sample = torch.tensor(np.array(target_sample))
        target_sample = target_sample.to(device)
        t_class = torch.tensor(t_class)
        t_class = t_class.to(device)
        #print(dir(model))
        target_layer = 'features.7'
        grad_cam = GradCAM(model, target_layer, device)
        triggers = np.zeros((tri_num, 28, 28, 1))
        for n in range(tri_num):
            img = target_sample[n].unsqueeze(0)  # [1, 28, 28, 1]
            input_tensor = img.detach().float().permute(0, 3, 1, 2)  # [1, 1, 28, 28]
            cam = grad_cam.generate_cam(input_tensor, t_class)
            cam = cam.squeeze()  # to array (28, 28)
            # cam = cam / cam.max()
            # print(cam.shape)
            save_img(cam, '2_cam.png')
            img = img.cpu().numpy()
            tailored_img = np.zeros((28, 28, 1))
            for i in range(28):
                for j in range(28):
                    for k in range(1):
                        tailored_img[i, j, k] = cam[i, j] * img[0, i, j, k]
            save_img(tailored_img, '3_tailored_img.png')
            image_dct = img_dct(img[0])
            tailored_img_dct = img_dct(tailored_img)
            for i in range(28):
                for j in range(28):
                    for k in range(1):
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

def gen_poi_sample_3S(x, trigger, p_d_ratio, p_c_r_thres):
    # trigger embedding
    # save_img(x.transpose(1, 2, 0), '5_clean_img.png')
    x_dct = img_dct(x)  # [28, 28, 1] -> [28, 28, 1]
    for i in range(28):
        for j in range(28):
            for k in range(1):
                if trigger[0][i][j][k] != 0:
                    x_dct[i][j][k] = x_dct[i][j][k] + p_d_ratio * (trigger[0][i][j][k] - x_dct[i][j][k])
    x_idct = img_idct(x_dct)  # [28, 28, 1] -> [28, 28, 1]
    # save_img(x_idct.transpose(1, 2, 0) / 255, '6_poisoned_img.png')
    # pixel restriction
    for i in range(28):
        for j in range(28):
            for k in range(1):
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
            x_p = X_train[n]  # 28, 28, 1
            # print(x_p.shape)
            # plt.imshow(x_p)
            # plt.savefig('clean_image.png')
            x_p = gen_poi_sample_3S(x_p, trigger, p_d_ratio, p_c_r_thres)
            # print(x_p.shape)
            # plt.imshow(x_p)
            # plt.savefig('p_image.png')
            X_train[n] = x_p  # [n, 28, 28, 1]
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
            X_p.append(gen_poi_sample_3S(X_test[i], trigger, p_d_ratio, p_c_r_thres))
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
    img_1 = img[:, :, 0]
    img_1_dct = dct(dct(img_1.T, type=2, norm='ortho').T, type=2, norm='ortho')
    img_dct = np.zeros((28, 28, 1))
    img_dct[:, :, 0] = img_1_dct
    return img_dct

def img_idct(img):
    for i in range(1):
        img[:, :, i] = idct(idct(img[:, :, i].T, norm='ortho').T, norm='ortho')
    return img

def train_3sattack(num_poi=500, f_s_thres=0.05, p_d_ratio=0.9, p_c_r_thres=20):
    print("load data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = load_data()  # X: [-1, 28, 28, 1]

    print("num_poi: {}".format(num_poi))
    triggers = extract_triggers(X_train, Y_train, t_class=5, tri_num=1, train=False, trig=True, f_s_thres=f_s_thres, device=device)

    print("calculate smr-ssim")
    from skimage.metrics import structural_similarity as ssim
    random_seed = np.random.randint(1000, 9999)
    np.random.seed(random_seed)
    np.random.shuffle(X_train)
    np.random.seed(random_seed)
    np.random.shuffle(Y_train)
    smr_ssim_sum = 0
    psnr_sun = 0
    for i in range(100):
        img1 = X_train[i]  # 28*28*1
        img1 = img1.astype(np.uint8)  # 28*28*1
        img2 = gen_poi_sample_3S(img1, triggers, p_d_ratio=p_d_ratio, p_c_r_thres=p_c_r_thres)
        #print(img1.shape, img2.shape)
        #print(img1.min(), img1.max(), img2.min(), img2.max())
        if img1.shape != img2.shape:
            img2 = Image.fromarray(img2).resize(img1.shape[::-1], Image.LANCZOS)
            img2 = np.array(img2)

        _, ssim_matrix = ssim(img1, img2, full=True, channel_axis=2, data_range=255)
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
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 1, 28, 28]
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
    model = MNIST_CNN().to(device)
    summary(model, input_size=(1, 28, 28))
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
            torch.save(model.state_dict(), os.path.join(model_path, "MNIST_poi_badnets.pt"))

    print("training finished")

train_3sattack()