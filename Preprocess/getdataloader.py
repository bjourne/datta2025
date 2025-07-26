from textwrap import fill
import torch
import os
import numpy as np
import random


from PIL import Image, ImageEnhance, ImageOps
from Preprocess.augment import Cutout, CIFAR10Policy
from torchvision import datasets, transforms
from torchvision.transforms import *
from torch.utils.data import DataLoader


# your own data dir
DIR = {
    'CIFAR10': '/tmp/data',
    'CIFAR100': '/tmp/data',
    'ImageNet': '/home/ubuntu/Residual-Encoded-Distillation-for-Aggressive-Pooling/imagenet_dataset'
}

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

def GetCifar10(batchsize, attack=False):
    trans_t = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        CIFAR10Policy(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)
    ])
    if attack:
        trans = Compose([ToTensor()])
    else:
        trans = Compose([
            ToTensor(),
            Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader




def GetCifar100(batchsize):

    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=8)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)

    return train_dataloader, test_dataloader
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader



def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)# sampler=train_sampler,

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)#, sampler=test_sampler)
    return train_dataloader, test_dataloader
