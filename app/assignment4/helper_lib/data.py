import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=128, num_workers=0, augment=True):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))

    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_tfms
    )

    test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_tfms
    )

    use_cuda = torch.cuda.is_available()
    pin = True if use_cuda else False

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_loader
