# Triple-Stealthy-Backdoor-Attack
Open source code for paper: "3S-attack: Triple Stealthy Backdoor Attack with Main Feature Against DNN model"

The datasets we have used to test the 3S-attack are MNIST, GTSRB, CIFAR10, CIFAR100, and Animal10. 
They are all open soursed datasets and are available on internet.
The MNIST, GTSRB, CIFAR10, and CIFAR100 datasets are available in pytorch, use the following command to download:

from torchvision import datasets
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

gtsrb_train = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
gtsrb_test = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)

cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

While the Animal10 dataset can be found in the followng wedsite:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

All experiments are performed using Python 3.11.0 and Pytorch 2.5.1+cu118.