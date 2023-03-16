import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

from dataset import LFWDataset, get_lfw_dataset
from model import SphereFace
from A_softmax_Loss import compute_loss
from train import train
from test import test

def main():
    # Hyperparameters
    num_epochs = 40
    batch_size = 128
    learning_rate = 0.00001
    m = 4  # Margin for the A-Softmax loss
    num_classes = 5749  # Set the number of classes to a fixed value

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the LFW dataset
    train_dataset, test_dataset = get_lfw_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model
    model = SphereFace(num_classes).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Train and test the model
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, m)
        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}')

        # Update learning rate according to the scheduler
        scheduler.step()

    accuracy = test(model, test_loader, device)
    print(f'Test accuracy: {accuracy:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), "sphereface.pth")

if __name__ == "__main__":
    main()
