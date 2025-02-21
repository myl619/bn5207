#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:12:36 2025

@author: linlin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, img_as_float, metrics
from scipy.signal import wiener

original = img_as_float(io.imread('NCCT.jpg', as_gray=True)) 
corrupted = img_as_float(io.imread('NCCT_motion_noise.jpg', as_gray=True))  
original = original.astype(np.float32)

# Wiener Filter Implementation
def wiener_filter(img, kernel_size=5):
    return wiener(img, kernel_size)

# Apply Wiener Filter
filtered_wiener = wiener_filter(corrupted, kernel_size=5)

def gamma_correction(image, gamma=1.2):
    image = np.clip(image, 0, 1)  # Ensure values are within [0,1]
    return image ** gamma  

gamma_value = 0.8  
filtered_wiener = gamma_correction(filtered_wiener, gamma=gamma_value)

# Transformer
transformer = img_as_float(io.imread('CS.jpg', as_gray=True)) #from transformer
transformer = transformer.astype(np.float32)

# PSNR and SSIM
def calculate_metrics(img1, img2, method_name):
    psnr_value = metrics.peak_signal_noise_ratio(img1, img2, data_range=1)
    ssim_value = metrics.structural_similarity(img1, img2, data_range=1)
    print(f'{method_name} PSNR: {psnr_value:.6f} dB')
    print(f'{method_name} SSIM: {ssim_value:.6f}')

calculate_metrics(original, filtered_wiener, "Wiener Filter")
calculate_metrics(original, transformer, "Transformer")

fig, axes = plt.subplots(2, 4, figsize=(18, 7))

def add_colorbar(im, ax):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gray Value")

im1 = axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('a) Original Image')
add_colorbar(im1, axes[0, 0])

im2 = axes[0, 1].imshow(corrupted, cmap='gray')
axes[0, 1].set_title('b) Corrupted Image')
add_colorbar(im2, axes[0, 1])

im3 = axes[0, 2].imshow(filtered_wiener, cmap='gray')
axes[0, 2].set_title('c) Restored (Wiener Filter)')
add_colorbar(im3, axes[0, 2])

im5 = axes[0, 3].imshow(transformer, cmap='gray')
axes[0, 3].set_title('e) Restored (Transformer)')
add_colorbar(im5, axes[0, 3])

axes[1, 0].hist(original.ravel(), bins=256, color='black', log=True)
axes[1, 0].set_title('f) Original Image Histogram')
axes[1, 0].grid(False)

axes[1, 1].hist(corrupted.ravel(), bins=256, color='black', log=True)
axes[1, 1].set_title('g) Corrupted Image Histogram')
axes[1, 1].grid(False)

axes[1, 2].hist(filtered_wiener.ravel(), bins=256, color='black', log=True)
axes[1, 2].set_title('h) Restored (Wiener Filter)')
axes[1, 2].grid(False)

axes[1, 3].hist(transformer.ravel(), bins=256, color='black', log=True)
axes[1, 3].set_title('j) Restored (Transformer)')
axes[1, 3].grid(False)

for i in range(4):  
    axes[0, i].axis('off')

plt.tight_layout()
plt.show()