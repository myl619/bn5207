#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:28:05 2025

@author: linlin
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, exposure


original = img_as_float(io.imread('NCCT.jpg', as_gray=True))  
corrupted = img_as_float(io.imread('NCCT_periodic_noise.jpg', as_gray=True)) 
original = original.astype(np.float32)
corrupted = corrupted.astype(np.float32)


period = 10 
amplitude = 0.5  
vertical_frequency = 1 / period  
rows, cols = corrupted.shape 

# Notch
notch_filter = np.ones((rows, cols))
for i in range(rows):
    notch_filter[i, :] = 1 - amplitude * np.cos(2 * np.pi * vertical_frequency * i)

notch_filtered = corrupted * notch_filter


offset = 1 - amplitude  
row_vector = np.arange(1, rows+1).reshape(-1, 1)
cos_vector = amplitude * (1 + np.cos(2 * np.pi * row_vector / period)) / 2 + offset
ripples_pattern = np.tile(cos_vector, (1, cols)) 


ripples_removed = np.divide(corrupted, ripples_pattern, where=ripples_pattern!=0)  


def gamma_correction(image, gamma=1.2):
    """对图像进行 Gamma 校正"""
    image = np.clip(image, 0, 1)  
    return image ** gamma 

gamma_value = 0.8  
ripples_corrected = gamma_correction(ripples_removed, gamma=gamma_value)


fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# 颜色条辅助函数
def add_colorbar(im, ax):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gray Value")


im1 = axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('a) Original Image')
add_colorbar(im1, axes[0, 0])

im2 = axes[0, 1].imshow(corrupted, cmap='gray')
axes[0, 1].set_title('b) Corrupted Image')
add_colorbar(im2, axes[0, 1])

im3 = axes[0, 2].imshow(ripples_removed, cmap='gray')
axes[0, 2].set_title('c) Restored (Notch Filter Applied)')
add_colorbar(im3, axes[0, 2])

im4 = axes[0, 3].imshow(ripples_corrected, cmap='gray')
axes[0, 3].set_title(f'd) Restored (Gamma Corrected, γ={gamma_value})')
add_colorbar(im4, axes[0, 3])


axes[1, 0].hist(original.ravel(), bins=256, color='black', log=True)
axes[1, 0].set_title('e) Original Image Histogram (Log Scale)')
axes[1, 0].grid(False)

axes[1, 1].hist(corrupted.ravel(), bins=256, color='black', log=True)
axes[1, 1].set_title('f) Corrupted Image Histogram (Log Scale)')
axes[1, 1].grid(False)

axes[1, 2].hist(ripples_removed.ravel(), bins=256, color='black', log=True)
axes[1, 2].set_title('g) Restored (Notch Filter Applied)')
axes[1, 2].grid(False)

axes[1, 3].hist(ripples_corrected.ravel(), bins=256, color='black', log=True)
axes[1, 3].set_title(f'h) Restored (Gamma Corrected, γ={gamma_value})')
axes[1, 3].grid(False)


for i in range(4):  
    axes[0, i].axis('off')

plt.tight_layout()
plt.show()