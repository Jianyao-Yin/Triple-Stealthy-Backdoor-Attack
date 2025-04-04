import torchvision
import torchvision.transforms as transforms
import numpy as np

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
X_train = np.array(train_dataset.data)
Y_train = np.array(train_dataset.targets)
X_test = np.array(test_dataset.data)
Y_test = np.array(test_dataset.targets)

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

np.savez("./data/train_dataset.npz", X=X_train, Y=Y_train)
np.savez("./data/test_dataset.npz", X=X_test, Y=Y_test)

print(1)