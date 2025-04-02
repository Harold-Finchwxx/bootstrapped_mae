import os
import shutil
import torchvision.datasets as datasets
import pickle

def prepare_cifar10():
    # CIFAR-10的类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建数据目录
    os.makedirs('./data/cifar10', exist_ok=True)
    
    # 下载CIFAR-10数据集
    print("Downloading CIFAR-10 dataset...")
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # 创建训练集目录
    train_dir = './data/cifar10/train'
    os.makedirs(train_dir, exist_ok=True)
    
    # 创建验证集目录（原测试集）
    val_dir = './data/cifar10/val'
    os.makedirs(val_dir, exist_ok=True)
    
    # 移动训练集数据
    print("Organizing training data...")
    for i in range(len(dataset_train)):
        img, label = dataset_train[i]
        class_dir = os.path.join(train_dir, classes[label])
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f'image_{i}.png'))
    
    # 移动验证集数据（原测试集）
    print("Organizing validation data...")
    for i in range(len(dataset_test)):
        img, label = dataset_test[i]
        class_dir = os.path.join(val_dir, classes[label])
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f'image_{i}.png'))
    
    print("Dataset preparation completed!")
    print(f"Training set size: {len(dataset_train)}")
    print(f"Validation set size: {len(dataset_test)}")
    print("\nClass directories:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

if __name__ == '__main__':
    prepare_cifar10()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""
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
"""