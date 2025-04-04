import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from PIL import Image

# 确保数据存放路径
os.makedirs("./data", exist_ok=True)

# 图像转换：缩放到 32×32，保持像素值 0~255（不归一化）
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Lambda(lambda img: np.array(img, dtype=np.uint8))  # 保持 uint8
])

# 下载 GTSRB 训练集和测试集
train_dataset = GTSRB(root="./data/gtsrb", split='train', transform=transform, download=True)
test_dataset = GTSRB(root="./data/gtsrb", split='test', transform=transform, download=True)

# 提取图像和标签
train_images = np.array([img for img, _ in train_dataset], dtype=np.uint8)  # 训练集样本
train_labels = np.array([label for _, label in train_dataset], dtype=np.int64)  # 训练集标签

test_images = np.array([img for img, _ in test_dataset], dtype=np.uint8)  # 测试集样本
test_labels = np.array([label for _, label in test_dataset], dtype=np.int64)  # 测试集标签

# 生成随机索引并打乱训练集
train_indices = np.random.permutation(len(train_images))
train_images = train_images[train_indices]
print(train_images.shape)
train_labels = train_labels[train_indices]

# 生成随机索引并打乱测试集
test_indices = np.random.permutation(len(test_images))
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]

# 保存为 .npz 格式
np.savez("./data/GTSRB_train.npz", X=train_images, Y=train_labels)
np.savez("./data/GTSRB_test.npz", X=test_images, Y=test_labels)

print("数据已成功处理并保存为 .npz 文件：")
print("- 训练集: ./data/GTSRB_train.npz")
print("- 测试集: ./data/GTSRB_test.npz")