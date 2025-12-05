import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from pytorch_msssim import SSIM

from model import UNet
from dataloaders import TrainLoader, ValLoader, TestLoader

torch.multiprocessing.set_sharing_strategy('file_system')

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

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_trainable}")

    print(" === MODEL SUMMARY === ")
    print(modelg)

    PATH = r"models\\attempt1\\model1.pth"

    def train():
        num_train = len(TrainLoader)
        num_val = len(ValLoader)
        num_test = len(TestLoader)
        num_epochs = 10
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        LRScheduler = scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=0)
        MSELoss = nn.MSELoss()
        SSIMLoss = SSIM(win_size=7, data_range=1.0, size_average=True, channel=1)

        for epoch in range(num_epochs):
            print(f" === EPOCH {epoch} === ")
            mse_loss = 0.0
            avg_snr = 0.0
            epoch_mse_train = 0.0

            modelg.train()
            for noisy, clean in TrainLoader:
                noisy=noisy.to(device)
                clean=clean.to(device)
                modelout = modelg(noisy)
                mse_loss = MSELoss(modelout, clean)
                avg_snr += calculate_snr(modelout, clean)

                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

                epoch_mse_train += mse_loss.item()

            print(f"Train MSE loss: {epoch_mse_train / num_train}")
            print(f"Average SNR over train set: {avg_snr / num_train}")

            print("\n")

            epoch_val_mse = 0.0
            epoch_val_snr = 0.0
            modelg.eval()
            with torch.no_grad():
                for noisy, clean in ValLoader:
                    noisy=noisy.to(device)
                    clean=clean.to(device)
                    modelout = modelg(noisy)
                    epoch_val_mse = MSELoss(modelout, clean).item()
                    epoch_val_snr += calculate_snr(modelout, clean)
            
                print(f"Val MSE Loss: {epoch_val_mse / num_val}")
                print(f"Average SNR over validation set: {epoch_val_snr / num_val}")

            print("\n")
            LRScheduler.step(epoch_val_mse / num_val)
            # break

        torch.save(model.state_dict(), PATH)

        epoch_test_mse = []
        mse_loss = 0.0
        epoch_test_snr = 0.0
        epoch_test_ssim = []
        ssim_loss = 0.0
        epoch_test_loss = 0.0
        modelg.eval()
        with torch.no_grad():
            for noisy, clean in TestLoader:
                noisy=noisy.to(device)
                clean=clean.to(device)
                modelout = modelg(noisy)
                mse_loss = MSELoss(modelout, clean).item()
                epoch_test_mse.append(mse_loss)
                ssim_loss = 1 - SSIMLoss(modelout, clean)
                epoch_test_ssim.append(ssim_loss)
                epoch_test_snr += calculate_snr(modelout, clean)
                epoch_test_loss += mse_loss + ssim_loss
        
            print(f"Test MSE Loss: {torch.sum(torch.tensor(epoch_test_mse)) / num_test}")
            print(f"Test SSIM Loss: {torch.sum(torch.tensor(epoch_test_ssim)) / num_test}")
            print(f"Average SNR over test set: {epoch_test_snr / num_test}")

    train()