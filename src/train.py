import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils import image_to_dataloaders, download_dataset
from model import CNN, LSTMModel
from plots import plot_learning_curve
from constants import *

def train_model(dataset_path, dataset_url, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the model with the given dataset.

    Parameters:
    - dataset_path: Path to the dataset.
    - dataset_url: URL to download the dataset if not found locally.
    - num_epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download the dataset if not found
    download_dataset(dataset_url, dataset_path)

    # Prepare data loaders for training and validation
    train_loader, val_loader = image_to_dataloaders(f'{dataset_path}/asl_alphabet_train/asl_alphabet_train', batch_size)

    # Initialize the model, loss function, and optimizer
    model, criterion, optimizer = initialize_model(device, learning_rate)
    
    # Run the training loop and collect training and validation losses
    train_losses, val_losses = run_training_loop(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
    # Plot the learning curves
    plot_learning_curve(train_losses, val_losses, save_path=MODEL_PATH)
    
    # Save the trained model
    save_model(model, MODEL_PATH)

def initialize_model(device, learning_rate):
    """
    Initialize the model, loss function, and optimizer.

    Parameters:
    - device: The device to use for computation.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model: The initialized model.
    - criterion: The loss function.
    - optimizer: The optimizer.
    """
    model = CNN(3, len(LABELS))
    model = model.to(device)  # Move the model to the specified device
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Define the optimizer
    return model, criterion, optimizer

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Run the training loop and collect training and validation losses.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: The loss function.
    - optimizer: The optimizer.
    - device: The device to use for computation.
    - num_epochs: Number of epochs for training.

    Returns:
    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate the model
        val_loss = validate_model(model, val_loader, criterion, device)
        
        # Record the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print the losses for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - criterion: The loss function.
    - optimizer: The optimizer.
    - device: The device to use for computation.

    Returns:
    - running_loss: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the parameters
        running_loss += loss.item() * images.size(0)  # Accumulate the loss

    return running_loss / len(train_loader.dataset)

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on the validation dataset.

    Parameters:
    - model: The model to validate.
    - val_loader: DataLoader for validation data.
    - criterion: The loss function.
    - device: The device to use for computation.

    Returns:
    - val_loss: The average validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            val_loss += loss.item() * images.size(0)  # Accumulate the loss
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

    val_loss /= len(val_loader.dataset)  # Compute the average loss
    accuracy = 100 * correct / total  # Compute the accuracy
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return val_loss

def save_model(model, model_path):
    """
    Save the trained model to disk.

    Parameters:
    - model: The trained model.
    - model_path: Path to save the model.
    """
    save_path = f"{model_path}/sign_language_model.pth"
    torch.save(model.state_dict(), save_path)  # Save the model state dictionary
    print(f"Model saved to {save_path}")

def test_model(model, test_loader, criterion, device):
    """
    Test the model on the test dataset.

    Parameters:
    - model: The model to test.
    - test_loader: DataLoader for test data.
    - criterion: The loss function.
    - device: The device to use for computation.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU
            labels = torch.argmax(labels, dim=1)  # Get the true class labels
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            test_loss += loss.item() * images.size(0)  # Accumulate the loss
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Compute the average loss
    accuracy = 100 * correct / total  # Compute the accuracy
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train_model(DATASET_PATH, DATASET_URL, num_epochs=25)