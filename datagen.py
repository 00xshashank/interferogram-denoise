import numpy as np
from tifffile import imsave
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)

N_IMAGES = 15000
N_SNR = N_IMAGES // 100
snr_db = np.linspace(-5, 25, N_SNR)

start = datetime.now()

print("Started generation...\n")

for snr in snr_db:
    baseIllumination = np.random.uniform(low=0.4, high=0.8, size=(N_SNR))
    gradXRand = np.random.uniform(low=-0.3, high=0.3, size=(N_SNR))
    gradX = gradXRand[:, None, None] * X[None, :, :]
    a = baseIllumination[:, None, None] + gradX
    b = (np.random.uniform(low=0.4, high=1.0, size=(N_SNR)))[:, None, None] - 0.3*(Y[None, :, :]**2)
    amp = np.random.uniform(low=0, high=10, size=(N_SNR))
    sigma = np.random.uniform(low=0.1, high=0.5, size=(N_SNR))
    bump = amp[:, None, None] * np.exp(-(X[None, :, :]**2 + Y[None, :, :]**2) / sigma[:, None, None])
    I = a + b*np.cos(bump)
    I_max_t = np.max(I, axis=(1,2))
    I_min_t = np.min(I, axis=(1,2))
    I_norm_t = (I_max_t - I_min_t) + 1e-8
    I_normalized_t = ((I - I_min_t[:, None, None]) / I_norm_t[:, None, None]) * 255.0

    for i in range(I_normalized_t.shape[0]):
        img = Image.fromarray(I_normalized_t[i].astype(np.uint8))
        img.save(f"data/target/target-{i}-snr-{snr}.png")

    awgn = np.random.normal(0, sigma[:, None, None], I.shape)
    signal_power = np.mean(I**2, axis=(1,2))
    noise_power_raw = np.mean(awgn**2, axis=(1,2))
    target_power = signal_power / (10**(snr/10))
    scale = np.sqrt(target_power / noise_power_raw)
    noise_scaled = awgn * scale[:, None, None]
    I += noise_scaled

    I_max = np.max(I, axis=(1,2))
    I_min = np.min(I, axis=(1,2))
    I_norm = (I_max - I_min) + 1e-8
    I_normalized = ((I - I_min[:, None, None]) / I_norm[:, None, None]) * 255.0

    for i in range(I_normalized.shape[0]):
        img = Image.fromarray(I_normalized[i].astype(np.uint8))
        img.save(f"data/image/image-{i}-snr-{snr}.png")

    print(f"\rFinished generation for SNR: {snr}\t\t\t\t", end=" ")

end = datetime.now()
print("\n\nTime taken: ", (end-start))