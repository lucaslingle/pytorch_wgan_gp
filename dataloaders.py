import torch as tc
import torchvision as tv

def get_celeba_dataloaders(batch_size):
    transform = tv.transforms.Compose([
        tv.transforms.CenterCrop(108),
        tv.transforms.Resize(64),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scales pixels to [-1, 1].
    ])
    train_data = tv.datasets.CelebA(root='data', split='train', download=True, transform=transform)
    test_data = tv.datasets.CelebA(root='data', split='test', download=True, transform=transform)

    train_dataloader = tc.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_cifar10_dataloaders(batch_size):
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scales pixels to [-1, 1].
    ])
    train_data = tv.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_data = tv.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    train_dataloader = tc.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_mnist_dataloaders(batch_size):
    transform = tv.transforms.Compose([
        tv.transforms.Resize(32),
        tv.transforms.ToTensor()
    ])
    train_data = tv.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = tv.datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_dataloader = tc.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader