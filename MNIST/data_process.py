import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 下载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

X_train = train_dataset.train_data.numpy()
X_train = np.expand_dims(X_train, axis=-1)
Y_train = train_dataset.train_labels.numpy()
X_test = test_dataset.test_data.numpy()
X_test = np.expand_dims(X_test, axis=-1)
Y_test = test_dataset.test_labels.numpy()
print(X_train.shape, Y_train.shape)
np.savez("./data/MNIST_train.npz", X=X_train, Y=Y_train)
np.savez("./data/MNIST_test.npz", X=X_test, Y=Y_test)

print(1)