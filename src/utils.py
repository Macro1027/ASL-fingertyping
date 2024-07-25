import os
import torch
import subprocess
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from PIL import Image

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Define the dataset class with one-hot encoding
class SignLanguageMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = np.array([self.data.iloc[idx, 1:].values.reshape(200, 200)])
        image = torch.tensor(image, dtype=torch.float32) / 255

        if self.transform:
            image = self.transform(image)


        # One-hot encode the label
        label = F.one_hot(torch.tensor(label), num_classes=25).float()

        return image, label

greyscale_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),  # Convert to grayscale
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # Convert the image to a tensor and scale to [0, 1]
    v2.Normalize((0.5,), (0.5,))  # Normalize the image to [-1, 1]
])

train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(200, 200), scale=(0.8, 1.0)),  # Randomly crop and resize
    v2.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
    v2.RandomRotation(degrees=15),  # Randomly rotate
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download kaggle sign language dataset
def download_dataset(dataset_url, dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    try:
        # Download the dataset using Kaggle API
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_url, '-p', dataset_path], check=True)
        # Unzip the dataset
        zip_path = os.path.join(dataset_path, 'asl-alphabet.zip')
        subprocess.run(['unzip', zip_path, '-d', dataset_path], check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def image_to_dataloaders(train_path, batch_size=64, num_workers=4):
    dataset = ImageFolder(train_path, transform=train_transform)

    train_ratio = 0.9
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# Convert raw images into dataloaders
def csv_to_dataloaders(dataset_path, batch_size=32):
    # Load the Sign Language MNIST dataset from csv

    train_dataset = SignLanguageMNISTDataset(os.path.join(dataset_path, 'sign_mnist_train.csv'), transform=train_transform)
    test_dataset = SignLanguageMNISTDataset(os.path.join(dataset_path, 'sign_mnist_test.csv'), transform=test_transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader