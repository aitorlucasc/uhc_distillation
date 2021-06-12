from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MnistDataset(Dataset):
    def __init__(self, data, target, transformation=None):
        self.images = data
        self.targets = target
        self.transformation = transforms.Compose([
            transforms.RandomAffine(0, (1 / 14, 1 / 14)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
