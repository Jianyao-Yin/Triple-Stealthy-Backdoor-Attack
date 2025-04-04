import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image


def crop_image_to_square(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        # 选择短边作为正方形边长
        square_size = width if width < height else height
        # 计算裁剪坐标
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        # 裁剪图片
        img = img.crop((left, top, right, bottom))
        img.save(image_path)
        img_array = np.array(img)
        return img_array


def read_directory_files(path):
    file_list = os.listdir(path)
    # for file in file_list:
    #     print(file)
    return file_list

input_Train_Path = './raw-img'
img_size = (128, 128)
X_train = []
Y_train = []
X_test = []
Y_test = []

f_train = read_directory_files(input_Train_Path)
for i in range(len(f_train)):
    ff_train = read_directory_files(('%s/%s' % (input_Train_Path, f_train[i])))
    for j in range(len(ff_train)):
        image_array = crop_image_to_square(('%s/%s/%s' % (input_Train_Path, f_train[i], ff_train[j])))
        image_resize = cv2.resize(src=image_array, dsize=img_size)

        if image_resize.shape == (128, 128, 3):
            X_train.append(image_resize)
        elif image_resize.shape == (128, 128, 4):
            # for ii in range(128):
            #     for jj in range(128):
            #         if image_resize[ii, jj, 3] == 0:
            #             image_resize[ii, jj] = [255, 255, 255, 0]
            image_resize = image_resize[:, :, :3]
            # plt.imshow(image_resize)
            # plt.show()
            X_train.append(image_resize)
        elif image_resize.shape == (128, 128):
            img_r = np.zeros((128, 128, 3))
            img_r[:, :, 0] = image_resize
            img_r[:, :, 1] = image_resize
            img_r[:, :, 2] = image_resize
            X_train.append(img_r)
        else:
            print(f"Image shape mismatch: {image_resize.shape}")
            break
            # plt.imshow(image_resize)
            # plt.show()
        Y_train.append(i)
        print(('%s/%s' % (i, j)))

random_seed = np.random.randint(1000, 9999)
# random_seed = 123
np.random.seed(random_seed)
np.random.shuffle(X_train)
np.random.seed(random_seed)
np.random.shuffle(Y_train)

X_test = X_train[:2500]
Y_test = Y_train[:2500]
X_train = X_train[2500:]
Y_train = Y_train[2500:]

random_seed = np.random.randint(1000, 9999)
# random_seed = 123
np.random.seed(random_seed)
np.random.shuffle(X_train)
np.random.seed(random_seed)
np.random.shuffle(Y_train)
np.random.seed(random_seed)
np.random.shuffle(X_test)
np.random.seed(random_seed)
np.random.shuffle(Y_test)

X_train = np.array(X_train, dtype=int)
Y_train = np.array(Y_train, dtype=int)
X_test = np.array(X_test, dtype=int)
Y_test = np.array(Y_test, dtype=int)

np.savez("./data/Animal10_train.npz", X=X_train, Y=Y_train)
np.savez("./data/Animal10_test.npz", X=X_test, Y=Y_test)

print(1)