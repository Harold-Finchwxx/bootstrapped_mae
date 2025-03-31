import torchvision.datasets as datasets
import pickle
"""
# 下载CIFAR-10数据集
dataset_train = datasets.CIFAR10(root='./data', train=True, download=True)
dataset_test = datasets.CIFAR10(root='./data', train=False, download=True)

print(dataset_train)
print(dataset_test)
"""

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('./data/cifar-10-batches-py/data_batch_1')
data1 = unpickle('./data/cifar-10-batches-py/batches.meta')
print(data.keys())
print(data1.keys())
print(data1)
print(data1[b'num_cases_per_batch'])
print(data1[b'label_names'])
print(data1[b'num_vis'])

print(data[b'data'][0])
print(data[b'labels'][0])
print(data[b'data'][0].shape)
print(data[b'labels'][0].shape)
