import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LFWDataset, get_lfw_dataset
from model import SphereFace
from A_softmax_Loss import compute_loss

def train(model, train_loader, optimizer, device, m):
    model.train()
    running_loss = 0.0
    for i, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()

        logits1 = model(img1)
        logits2 = model(img2)

        loss = compute_loss(logits1, label, m) + compute_loss(logits2, label, m)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping

        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

