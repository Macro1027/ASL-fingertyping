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
    # Download the dataset if not found
    download_dataset(dataset_url, dataset_path)

    train_path = f'{dataset_path}/asl_alphabet_train/asl_alphabet_train'
    train_loader, val_loader = image_to_dataloaders(train_path)

    # Instantiate the model, define the loss function and the optimizer
    model = vgg19(weights=VGG19_Weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(LABELS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    val_losses = []
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Record losses
        train_losses.append(running_loss)
        val_loss = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # test_model(model, test_loader, criterion)

    # Plot losses
    plot_learning_curve(train_losses, val_losses, save_path=MODEL_PATH)

    # Save the model
    print(f"Model saved to {MODEL_PATH}/sign_language_model.pth")
    torch.save(model.state_dict(), f"{MODEL_PATH}/sign_language_model.pth")

def validate_model(model, val_loader, criterion):
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    return val_loss

def test_model(model, test_loader, criterion):
    # Test loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = torch.argmax(labels, dim=1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_model(DATASET_PATH, DATASET_URL, num_epochs=25)