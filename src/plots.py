import matplotlib.pyplot as plt
from utils import image_to_dataloaders, download_dataset, unnormalize
import torch
from constants import *

def plot_images_from_loader(loader, classes, n=5):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Plot the images
    fig = plt.figure(figsize=(15, 5))
    for idx in range(n):
        ax = fig.add_subplot(1, n, idx+1, xticks=[], yticks=[])
        img = images[idx].permute((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = unnormalize(img)
        plt.imshow(img, cmap='gray')
        label = torch.argmax(labels[idx]).item()
        ax.set_title(classes[label])
    plt.show()

def plot_learning_curve(train_losses, val_losses, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')  # Blue line for training loss
    plt.plot(epochs, val_losses, 'orange', label='Validation Loss')  # Orange line for validation loss

    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Download the dataset if not found
    download_dataset(DATASET_PATH, DATASET_URL)

    # Load data
    _, val_loader, = image_to_dataloaders(DATASET_PATH)
    plot_images_from_loader(val_loader, LABELS)