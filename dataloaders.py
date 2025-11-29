import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.io import read_image
from torchvision import transforms

import os

data_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32)
])

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
        fnoisy_img = noisy_img.float()
        tname = fname.replace("image", "target")
        target_img = read_image(f"{self.targets_dir}\\{tname}")
        ftarget_img = target_img.float()
        if self.transform is not None:
            fnoisy_img = self.transform(noisy_img)
            ftarget_img = self.transform(target_img)

        return fnoisy_img, ftarget_img

    def __len__(self):
        return len(os.listdir(f"data\\{self.split}\\images"))    

TrainDataset = ImageDataset("train", transform=data_transform)
TrainLoader = data.DataLoader(
    TrainDataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

ValDataset = ImageDataset("val", transform=data_transform)
ValLoader = data.DataLoader(
    ValDataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

TestDataset = ImageDataset("test", transform=data_transform)
TestLoader = data.DataLoader(
    TestDataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

if __name__ == "__main__":
    for noisy, clean in TrainLoader:
        print(f"Shape of noisy: {noisy.shape}, clean: {clean.shape}")
        print(f"Max values: {torch.max(noisy)}, {torch.max(clean)}, min values: {torch.min(noisy)}, {torch.min(clean)}")
        import matplotlib.pyplot as plt
        plt.imshow(clean[1][0], cmap='gray')
        plt.show()
        break