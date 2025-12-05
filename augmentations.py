import torch
import torch.nn as nn
from torchvision.io import read_image

import random

class Normalize(nn.Module):
    def __init__(self, upper_val = 255.0):
        super().__init__()
        self.upper_val = upper_val

    @torch.no_grad()
    def forward(self, image: torch.tensor, target: torch.tensor):
        return (
            image.float() / self.upper_val,
            target.float() / self.upper_val
        )
    
    def __repr__(self):
        return f"Divide all pixels in image by: {self.upper_val}"

class RotateImage(nn.Module):
    def __init__(self, prob=0.25, dims=(-2, -1)):
        super().__init__()
        self.prob = prob
        self.dims = dims

    @torch.no_grad()
    def forward(self, image, target):
        if random.random() > self.prob:
            return image, target

        return (
            torch.rot90(image, 1, self.dims),
            torch.rot90(target, 1, self.dims),
        )
    
    def __repr__(self):
        return f"Rotate image with around dims: {self.dims} probability: {self.prob}"


class RollImage(nn.Module):
    def __init__(self, prob=0.25, shifts=50):
        super().__init__()
        self.prob = prob
        self.shifts = shifts

    @torch.no_grad()
    def forward(self, image, target):
        if random.random() > self.prob:
            return image, target

        return (
            torch.roll(image, shifts=self.shifts),
            torch.roll(target, shifts=self.shifts),
        )
    
    def __repr__(self):
        return f"Roll image with shifts: {self.shifts}"

class Transforms(nn.Module):
    def __init__(self, modules: list[nn.Module] = [
        RotateImage(),
        RollImage(),
        Normalize()
    ]):
        super().__init__()
        self.transforms = nn.ModuleList([*modules])

    def forward(self, image, target):
        for module in self.transforms:
            image, target = module(image, target)

        return image, target

if __name__ == "__main__":
    def main():
        image = read_image(r"data\\test\\images\\image-3-snr-22.18120805369127.png")
        target = read_image(r"data\\test\\labels\\target-3-snr-22.18120805369127.png")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)

        transforms = Transforms()

        axes[0].imshow(image[0], cmap='gray')
        axes[0].set_title("Original Image")

        timg, tlabel = transforms(image, target)
        print(f"Shape if timg: {timg.shape}, tlabel: {tlabel.shape}")
        print(f"Max value in image: {torch.max(timg)}, label: {torch.max(tlabel)}")
        axes[1].imshow(timg[0], cmap='gray')
        axes[1].set_title("Rotated Image")

        plt.show()

    main()