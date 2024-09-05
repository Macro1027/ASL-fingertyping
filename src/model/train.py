import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ultralytics import YOLO

# Add the source directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

# Import custom modules
from model.utils import image_to_dataloaders, download_dataset, load_dataset
from model.model import CNN, LSTMModel
from model.plots import plot_learning_curve
from constants import *

class CustomModel:
    def __init__(self):
        pass

    def train_model(self, dataset_path, dataset_url, num_epochs=10, batch_size=64, learning_rate=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device}.")

        # Prepare data loaders for training and validation
        train_loader, val_loader = image_to_dataloaders(
            f'{dataset_path}/asl_alphabet_train/asl_alphabet_train', batch_size)

        # Initialize the model, loss function, and optimizer
        model, criterion, optimizer = self.initialize_model(device, learning_rate)

        # Run the training loop and collect training and validation losses
        train_losses, val_losses = self.run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

        # Plot the learning curves
        plot_learning_curve(train_losses, val_losses, save_path=MODEL_PATH)

        # Save the trained model
        self.save_model(model, MODEL_PATH)

    def initialize_model(self, device, learning_rate):
        model = CNN(3, len(LABELS)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return model, criterion, optimizer

    def run_training_loop(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = self.validate_model(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        return train_losses, val_losses

    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        return running_loss / len(train_loader.dataset)

    def validate_model(self, model, val_loader, criterion, device):
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return val_loss

    def save_model(self, model, model_path):
        save_path = f"{model_path}/sign_language_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

class YOLOv9:
    def __init__(self):
        pass

    def train(self):
        model = YOLO("yolov9c.pt")
        results = model.train(data="data/American-Sign-Language-Letters-1/data.yaml", epochs=100, imgsz=640)
        print("done training")

if __name__ == "__main__":
    model = YOLOv9()
    model.train()