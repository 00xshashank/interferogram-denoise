import torch
import torch.utils.data as data
from torchvision.io import read_image

import os

from augmentations import Transforms

class ImageDataset(data.Dataset):
    def __init__(self, split, transform=None):
        super().__init__()
        self.split = split
        self.image_dir = f"data\\{self.split}\\images"
        self.targets_dir = f"data\\{self.split}\\labels"
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __getitem__(self, index):
        fname = self.image_files[index]
        noisy_img = read_image(f"{self.image_dir}\\{fname}")
        tname = fname.replace("image", "target")
        target_img = read_image(f"{self.targets_dir}\\{tname}")
        if self.transform is not None:
            return self.transform(noisy_img, target_img)

        return noisy_img, target_img

    def __len__(self):
        return len(os.listdir(f"data\\{self.split}\\images"))    

TrainDataset = ImageDataset("train", transform=Transforms())
ValDataset = ImageDataset("val", transform=Transforms())
TestDataset = ImageDataset("test", transform=Transforms())
    
TrainLoader = data.DataLoader(
    TrainDataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
ValLoader = data.DataLoader(
    ValDataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
TestLoader = data.DataLoader(
    TestDataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

if __name__ == "__main__":
    for noisy, clean in TrainLoader:
        print(f"Shape of noisy: {noisy.shape}, clean: {clean.shape}")
        print(f"Max values: {torch.max(noisy)}, {torch.max(clean)}, min values: {torch.min(noisy)}, {torch.min(clean)}")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(noisy[1][0], cmap='gray')
        axes[0].set_title("Noisy Image")
        axes[1].imshow(clean[1][0], cmap='gray')
        axes[1].set_title("Clean Image")
        plt.show()
        break