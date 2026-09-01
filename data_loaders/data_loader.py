import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os

from data_loaders import cub_loader  

from data_loaders.AwA_loaders import load_CIFAR_AwA_data, load_AwA_data_all, load_AwA_data_17
from data_loaders.aPY_loaders import load_CIFAR_data_apy, load_aPY_data
from data_loaders.AwA_load import load_AwA_2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def get_transforms(dataset_name, save_dir, image_size=224, normalize=True):
    
    resol = image_size
    
    if 'cifar' in dataset_name:
        mean, std = CIFAR_MEAN, CIFAR_STD
        
        resized_resol = int(resol * 256 / 224)
        trainTransform = transforms.Compose([transforms.ColorJitter(brightness=32 / 255, saturation=0.5),
                                            transforms.Resize((resized_resol, resized_resol)),
                                            transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])


    elif "awa2" in dataset_name:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        
        resized_resol = int(resol * 256 / 224)  
        trainTransform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.1, saturation=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])
                    
    
    elif "apy" in dataset_name:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        
        resized_resol = int(resol * 299 / 224)
        trainTransform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.1, saturation=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])

        
        
    
    
    testTransform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])


    """transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]"""
    """if normalize:
        transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'transforms_used.txt'), 'w') as f:
            f.write('Train Transformations:\n')
            f.write(str(trainTransform) + '\n\n')
            f.write('Test Transformations:\n')
            f.write(str(testTransform) + '\n')
        
    return trainTransform, testTransform

def get_standard_dataset(name, config, seed, save_dir, bs, data_dir):
    
    train_transform, test_transform = get_transforms(name, save_dir, config.get("image_size"), normalize=True)

    val_split = config.get("val_split")
    
    if config.get("batch_size") is None:
        batch_size = bs
    else:
        batch_size = config.get("batch_size")
    
    #data_dir = config.get("data_dir")
    
    # Defaults
    imbalance = None
    concept_group_map = {}
    concepts = []
    classes = []

    if name == "cifar10":
        
        full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        classes = full_train.classes  # List of class names

        # Split into train/val
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_set, val_set = random_split(full_train, [train_size, val_size])

        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    elif name == "mnist":
        
        full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        classes = [str(i) for i in range(10)]  # MNIST digits as strings

        # Split into train/val
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_set, val_set = random_split(full_train, [train_size, val_size])

        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    elif name == "cifar100_awa2":
        
        X_train, X_test, y_train, y_test, a_train, a_test, mat_pd, classes, concepts, label_to_idx_awa, GT_matrix = \
            load_CIFAR_AwA_data(False, config)


        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_train, y_train, a_train, test_size=val_split, random_state=seed, stratify=y_train)

        def make_tensor_loader(X, Y, C, transform, shuffle):
            X_ = [transform(Image.fromarray((x * 255).astype(np.uint8))) for x in X]
            x_tensor = torch.stack(X_)
            y_tensor = torch.tensor(Y.squeeze(), dtype=torch.long)
            c_tensor = torch.tensor(C, dtype=torch.float)
            dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_dl = make_tensor_loader(X_train, y_train, a_train, train_transform, shuffle=True)
        val_dl = make_tensor_loader(X_val, y_val, a_val, test_transform, shuffle=False)
        test_dl = make_tensor_loader(X_test, y_test, a_test, test_transform, shuffle=False)


    elif name == "awa2":
        X_train, X_test, y_train, y_test, a_train, a_test, mat_pd, classes, concepts, label_to_idx_awa, GT_matrix = \
            load_AwA_data_all(False, config)


        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_train, y_train, a_train, test_size=val_split, random_state=seed, stratify=y_train)

        def make_tensor_loader(X, Y, C, transform, shuffle):
            X_ = [transform(Image.fromarray((x).astype(np.uint8))) for x in X]
            x_tensor = torch.stack(X_)
            y_tensor = torch.tensor(Y.squeeze(), dtype=torch.long)
            c_tensor = torch.tensor(C, dtype=torch.float)
            dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_dl = make_tensor_loader(X_train, y_train, a_train, train_transform, shuffle=True)
        val_dl = make_tensor_loader(X_val, y_val, a_val, test_transform, shuffle=False)
        test_dl = make_tensor_loader(X_test, y_test, a_test, test_transform, shuffle=False)
        
        
    elif name == "awa2_cv":
        images, _, labels, _, annotations, _, mat_pd_awa, classes, concepts, label_to_idx_awa, mat_GT = \
            load_AwA_2(True, data_dir)

        return images, _, labels, _, annotations, _, mat_pd_awa, classes, concepts, label_to_idx_awa, mat_GT, train_transform, test_transform
    

        
    elif name == "awa2_17":
        X_train, X_test, y_train, y_test, a_train, a_test, mat_pd, classes, concepts, label_to_idx_awa, GT_matrix = \
            load_AwA_data_17(False, data_dir)


        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_train, y_train, a_train, test_size=val_split, random_state=seed, stratify=y_train)

        def make_tensor_loader(X, Y, C, transform, shuffle):
            X_ = [transform(Image.fromarray((x * 255).astype(np.uint8))) for x in X]
            x_tensor = torch.stack(X_)
            y_tensor = torch.tensor(Y.squeeze(), dtype=torch.long)
            c_tensor = torch.tensor(C, dtype=torch.float)
            dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_dl = make_tensor_loader(X_train, y_train, a_train, train_transform, shuffle=True)
        val_dl = make_tensor_loader(X_val, y_val, a_val, test_transform, shuffle=False)
        test_dl = make_tensor_loader(X_test, y_test, a_test, test_transform, shuffle=False)
        
    elif name == "apy":
        X_train, X_test, y_train, y_test, a_train, a_test, mat_pd, classes, concepts, label_to_idx_awa, GT_matrix = \
            load_aPY_data(False, data_dir, config)


        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_train, y_train, a_train, test_size=val_split, random_state=seed, stratify=y_train)

        def make_tensor_loader(X, Y, C, transform, shuffle):
            X_ = [transform(Image.fromarray((x * 255).astype(np.uint8))) for x in X]
            x_tensor = torch.stack(X_)
            y_tensor = torch.tensor(Y.squeeze(), dtype=torch.long)
            c_tensor = torch.tensor(C, dtype=torch.float)
            dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_dl = make_tensor_loader(X_train, y_train, a_train, train_transform, shuffle=True)
        val_dl = make_tensor_loader(X_val, y_val, a_val, test_transform, shuffle=False)
        test_dl = make_tensor_loader(X_test, y_test, a_test, test_transform, shuffle=False)
        
    
    elif name == "apy_cv":
        images, _, labels, _, annotations, _, mat_pd_awa, classes, concepts, label_to_idx_awa, mat_GT = \
             load_aPY_data(True, data_dir, config)

        return images, _, labels, _, annotations, _, mat_pd_awa, classes, concepts, label_to_idx_awa, mat_GT, train_transform, test_transform
    
        
    
    elif name == "cifar100_apy":
        X_train, X_test, y_train, y_test, a_train, a_test, mat_pd, classes, concepts, label_to_idx_awa, GT_matrix = \
            load_CIFAR_data_apy(False, config)


        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_train, y_train, a_train, test_size=val_split, random_state=seed, stratify=y_train)

        def make_tensor_loader(X, Y, C, transform, shuffle):
            X_ = [transform(Image.fromarray((x * 255).astype(np.uint8))) for x in X]
            x_tensor = torch.stack(X_)
            y_tensor = torch.tensor(Y.squeeze(), dtype=torch.long)
            c_tensor = torch.tensor(C, dtype=torch.float)
            dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_dl = make_tensor_loader(X_train, y_train, a_train, train_transform, shuffle=True)
        val_dl = make_tensor_loader(X_val, y_val, a_val, test_transform, shuffle=False)
        test_dl = make_tensor_loader(X_test, y_test, a_test, test_transform, shuffle=False)


    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_dl, val_dl, test_dl, imbalance, concepts, classes, concept_group_map, label_to_idx_awa


def get_dataset_loaders(dataset_name, config, seed, save_dir, bs, data_dir, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == "cub":
        # returns: train_dl, val_dl, test_dl, imbalance, (n_concepts, N_CLASSES, concept_group_map)
        return cub_loader.generate_data(config, data_dir)

    else:
        return get_standard_dataset(dataset_name, config, seed, save_dir, bs, data_dir)
