import torchvision.datasets as ds
import torch.utils.data

import torchvision.transforms as transforms

import torchvision.transforms.transforms as transforms

import numpy as np
import torchvision

from enum import Enum


class DSNames(Enum):
    """Names for every Datasets

    """

    SVHN = "SVHN"
    MNIST = "MNIST"


class DSStastic:
    """Mean and Std for very Datasets
    """

    SVHN = np.array(([0.44, 0.44, 0.44], [0.19, 0.19, 0.19]))

    MNIST = np.array(([0.44, 0.44, 0.44], [0.19, 0.19, 0.191]))


def load_dataset(
    name: DSNames,
    batch_size,
    root="./data",
    split="train",
    download=False,
    mode="norm",
    size=224,
):
    """Helpper function to get `DataLoader` of specific datasets 
    
    Arguments:
        name {DSNames} -- data set
        param {[type]} -- parameters to generate dataloader
            - `param.batch_size` to spectic batch size.
    
    Keyword Arguments:
        root {str} -- [root path for dataset] (default: {'./data'})
        split {str} -- [get 'train' or 'valid' part] (default: {'train'})
        dowloard {bool} -- [if not exits, want to dowload?] (default: {False})
        mode {str} -- ['norm' the dataset or 'rerange' to [-1,1]] (default: {'norm'})
    
    Returns:
        [DataLoader] -- [a DataLoader for the dataset]
    """

    dsname = name.value
    mean_std = getattr(DSStastic, dsname)

    mean = mean_std[0]
    std = mean_std[1]

    trans = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if dsname == "MNIST":
        trans.insert(0, transforms.Grayscale(3))

    transform = transforms.Compose(trans)

    try:
        data_set = getattr(ds, dsname)(
            root="./_PUBLIC_DATASET_", split=split, transform=transform, download=True
        )
    except:
        train = split is "train"
        data_set = getattr(ds, dsname)(
            root="./_PUBLIC_DATASET_", train=train, transform=transform, download=True
        )

    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=batch_size, shuffle=True
    )

    return data_set, data_loader


def load_img_dataset(dataset, subset, batch_size):

    trans = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        ),
    ]
    transform = transforms.Compose(trans)

    dataset = ds.ImageFolder(
        root="./_PUBLIC_DATASET_/" + dataset + "/" + subset,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        # shuffle=True, 
        drop_last=True, 
    )

    return dataset, data_loader


if __name__ == "__main__":
    OH_art = ds.ImageFolder("_PUBLIC_DATASET_/OfficeHome/Art")
    OH_Clipart = ds.ImageFolder("_PUBLIC_DATASET_/OfficeHome/Art")
    print(OH_art.classes == OH_Clipart.classes)
    print(OH_Clipart.class_to_idx)

