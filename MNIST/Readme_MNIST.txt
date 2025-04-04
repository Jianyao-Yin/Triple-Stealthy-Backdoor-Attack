Download dataset and storage them in data folder:

from torchvision import datasets
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


After downloading the dataset, run data_process.py to pre-process the train and test dataset, then run train_3sattack_torch.py to train backdoor model

Pre-processed datasets are in npz format where 'x' is each sample and 'y' is their labels