#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:12:36 2025

@author: linlin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage import io, img_as_float, metrics, restoration
from skimage.filters import rank
from skimage.morphology import disk
from scipy.signal import wiener

def adaptive_median_filter(image, max_window_size=24):
    filtered_image = np.copy(image)
    height, width = image.shape
    
    for i in range(height):
        for j in range(width):
            window_size = 2
            while window_size <= max_window_size:
                i_min = max(i - window_size // 2, 0)
                i_max = min(i + window_size // 2 + 1, height)
                j_min = max(j - window_size // 2, 0)
                j_max = min(j + window_size // 2 + 1, width)
                
                window = image[i_min:i_max, j_min:j_max]
                median_value = np.median(window)
                min_value = np.min(window)
                max_value = np.max(window)
                
                if min_value < median_value < max_value:
                    if min_value < image[i, j] < max_value:
                        filtered_image[i, j] = image[i, j]
                    else:
                        filtered_image[i, j] = median_value
                    break
                else:
                    window_size += 2
    
    return filtered_image

def wavelet_denoising(image, wavelet='db2', level=2):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1][-1])) / 0.6745  


    coeffs_thresholded = [coeffs[0]]  
    for detail_coeffs in coeffs[1:]:
        coeffs_thresholded.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in detail_coeffs))
    
    return pywt.waverec2(coeffs_thresholded, wavelet)

original = img_as_float(io.imread('CCTA.jpg', as_gray=True)) 
corrupted = img_as_float(io.imread('CCTA_salt_and_pepper.jpg', as_gray=True))  
original = original.astype(np.float32)

# Apply Adaptive Median Filter
filtered_adaptive_median = adaptive_median_filter(corrupted)

# Apply Wavelet Denoising
filtered_wavelet = wavelet_denoising(corrupted)

def gamma_correction(image, gamma=1.2):
    image = np.clip(image, 0, 1)  # Ensure values are within [0,1]
    return image ** gamma  

gamma_value = 0.8  
filtered_adaptive_median = gamma_correction(filtered_adaptive_median, gamma=gamma_value)
filtered_wavelet = gamma_correction(filtered_wavelet, gamma=gamma_value)

# Transformer result
transformer = img_as_float(io.imread('CS.jpg', as_gray=True)) 
transformer = transformer.astype(np.float32)
transformer_after_adaptive_median = adaptive_median_filter(transformer)

def calculate_metrics(img1, img2, method_name):
    psnr_value = metrics.peak_signal_noise_ratio(img1, img2, data_range=1)
    ssim_value = metrics.structural_similarity(img1, img2, data_range=1)
    print(f'{method_name} PSNR: {psnr_value:.6f} dB')
    print(f'{method_name} SSIM: {ssim_value:.6f}')

calculate_metrics(original, filtered_adaptive_median, "Adaptive Median Filter")
calculate_metrics(original, filtered_wavelet, "Wavelet Denoising")
calculate_metrics(original, transformer, "Transformer")
calculate_metrics(original, transformer_after_adaptive_median, "transformer_after_adaptive_median")

fig, axes = plt.subplots(2, 6, figsize=(26, 8))

def add_colorbar(im, ax):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gray Value")

im1 = axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('a) Original Image')
add_colorbar(im1, axes[0, 0])

im2 = axes[0, 1].imshow(corrupted, cmap='gray')
axes[0, 1].set_title('b) Corrupted Image')
add_colorbar(im2, axes[0, 1])

im3 = axes[0, 2].imshow(filtered_adaptive_median, cmap='gray')
axes[0, 2].set_title('c) Restored (Adaptive Median)')
add_colorbar(im3, axes[0, 2])

im4 = axes[0, 3].imshow(filtered_wavelet, cmap='gray')
axes[0, 3].set_title('d) Restored (Wavelet)')
add_colorbar(im4, axes[0, 3])

im5 = axes[0, 4].imshow(transformer, cmap='gray')
axes[0, 4].set_title('e) Restored (Transformer)')
add_colorbar(im5, axes[0, 4])

im6 = axes[0, 5].imshow(transformer_after_adaptive_median, cmap='gray')
axes[0, 5].set_title('f) Restored (transformer_x)')
add_colorbar(im6, axes[0, 5])

axes[1, 0].hist(original.ravel(), bins=256, color='black', log=True)
axes[1, 0].set_title('g) Original Image Histogram')
axes[1, 0].grid(False)

axes[1, 1].hist(corrupted.ravel(), bins=256, color='black', log=True)
axes[1, 1].set_title('h) Corrupted Image Histogram')
axes[1, 1].grid(False)

axes[1, 2].hist(filtered_adaptive_median.ravel(), bins=256, color='black', log=True)
axes[1, 2].set_title('i) Restored (Adaptive Median)')
axes[1, 2].grid(False)

axes[1, 3].hist(filtered_wavelet.ravel(), bins=256, color='black', log=True)
axes[1, 3].set_title('j) Restored (Wavelet)')
axes[1, 3].grid(False)

axes[1, 4].hist(transformer.ravel(), bins=256, color='black', log=True)
axes[1, 4].set_title('k) Restored (Transformer)')
axes[1, 4].grid(False)

axes[1, 5].hist(transformer_after_adaptive_median.ravel(), bins=256, color='black', log=True)
axes[1, 5].set_title('l) Restored (transformer_x)')
axes[1, 5].grid(False)

for i in range(6):  
    axes[0, i].axis('off')

plt.tight_layout()
plt.show()
