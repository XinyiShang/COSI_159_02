import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import tarfile


class LFWDataset(Dataset):

    def __init__(self, root_dir, pairs_file, transform=None):
        self.root_dir = os.path.join(root_dir, 'lfw')
        self.transform = transform

        with open(pairs_file, 'r') as file:
            lines = file.readlines()[1:]  # Skip the first line
            self.pairs = [line.strip().split('\t') for line in lines]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        if len(pair) == 3:  # Positive pair
            name1, name2 = pair[0], pair[0]
            idx1, idx2 = int(pair[1]), int(pair[2])
            label = 1
        else:  # Negative pair
            name1, name2 = pair[0], pair[2]
            idx1, idx2 = int(pair[1]), int(pair[3])
            label = 0

        img1_path = os.path.join(self.root_dir, name1, f'{name1}_{idx1:04d}.jpg')
        img2_path = os.path.join(self.root_dir, name2, f'{name2}_{idx2:04d}.jpg')

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.long)

#code and dataset in the same folder
def get_lfw_dataset(input_size=96):
    root_dir = os.getcwd()
    
    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    pairs_train_file = os.path.join(root_dir, 'pairsDevTrain.txt')
    pairs_test_file = os.path.join(root_dir, 'pairsDevTest.txt')

    train_dataset = LFWDataset(root_dir, pairs_train_file, transform=data_transform)
    test_dataset = LFWDataset(root_dir, pairs_test_file, transform=data_transform)

    return train_dataset, test_dataset


def extract_lfw_dataset():
    root_dir = os.getcwd()
    lfw_tgz = os.path.join(root_dir, 'lfw.tgz')

    with tarfile.open(lfw_tgz, 'r:gz') as tar:
        tar.extractall(path=root_dir)

extract_lfw_dataset()
