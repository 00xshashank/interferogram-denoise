from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import UNet
from dataloaders import TrainLoader, ValLoader, TestLoader

def calculate_snr(output: torch.tensor, target: torch.tensor):
    output_flat = output.view(output.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    noise = output_flat - target_flat
    output_power = torch.mean(target_flat.pow(2), dim=1)
    noise_power = torch.mean(noise.pow(2), dim=1)

    power_noise = torch.where(noise_power==0, torch.tensor(1e-8, device=noise_power.device), noise_power)
    power_output = torch.where(output_power==0, torch.tensor(1e-8, device=output_power.device), output_power)

    return (10*torch.log(power_output / power_noise)).mean().item()

if __name__ == "__main__":
    model = UNet(
        input_dim=256,
        output_dim=256,
        channels=[3, 9, 27]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    modelg = model.to(device=device)

    print(" ===== MODEL SUMMARY ===== ")
    print(modelg)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_trainable}")

    PATH = r"models\\model1.pth"

    def train():
        num_train = len(TrainLoader)
        num_val = len(ValLoader)
        num_epochs = 10
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        CELoss = nn.MSELoss()

        train_losses = []
        for epoch in range(num_epochs):
            print(f" ====== EPOCH {epoch} ====== ")
            
            epoch_loss = 0.0
            avg_snr = 0.0

            modelg.train()
            for noisy, clean in TrainLoader:
                noisy=noisy.to(device)
                clean=clean.to(device)
                modelout = modelg(noisy)
                loss = CELoss(modelout, clean)
                epoch_loss += loss.item()
                avg_snr += calculate_snr(modelout, clean)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Train Loss: {epoch_loss / num_train}")
            print(f"Average SNR over train set: {avg_snr / num_train}")

            epoch_val_loss = 0.0
            epoch_val_snr = 0.0
            modelg.eval()
            with torch.no_grad():
                for noisy, clean in ValLoader:
                    noisy=noisy.to(device)
                    clean=clean.to(device)
                    modelout = modelg(noisy)
                    loss = CELoss(modelout, clean)
                    epoch_val_loss += loss.item()
                    epoch_val_snr += calculate_snr(modelout, clean)
            
                print(f"Val Loss: {epoch_val_loss / num_val}")
                print(f"Average SNR over validation set: {epoch_val_snr / num_val}")

            print("\n")
            # break

        torch.save(model.state_dict(), PATH)
            
        # plt.plot(train_losses)
        # plt.show()

    train()