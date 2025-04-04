Download dataset and storage them in data folder:

from torchvision import datasets
cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


After downloading the dataset, run data_process.py to pre-process the train and test dataset, then run train_3sattack_torch.py to train backdoor model

Pre-processed datasets are in npz format where 'x' is each sample and 'y' is their labels