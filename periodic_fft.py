import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, img_as_float, metrics


original = img_as_float(io.imread('CCTA.jpg', as_gray=True)) 
corrupted = img_as_float(io.imread('CCTA_motion_noise.jpg', as_gray=True))  
original = original.astype(np.float32)


def fft_filter(img):
    f = np.fft.fft2(img)  
    fshift = np.fft.fftshift(f)  
    magnitude_spectrum = np.log(1 + np.abs(fshift))  

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2 
    mask = np.ones((rows, cols), np.uint8)

    mask[crow-30:crow+30, ccol-70:ccol+70] = 0  
    rows, cols = corrupted.shape
    crow, ccol = rows // 2 , cols // 2  # center
    

    mask = np.ones((rows, cols), np.uint8)

    notch_points = [
        (209, 256),  # point1
        # (255, 257)  # point2

    ]

    a = 12
    for (r, c) in notch_points:
        mask[r-a:r+a, c-a:c+a] = 0  
        mask[rows-r-a:rows-r+a, cols-c-a:cols-c+a] = 0  
    #mask
    fshift_filtered = fshift * mask  
    f_ishift = np.fft.ifftshift(fshift_filtered)  
    img_filtered = np.fft.ifft2(f_ishift)  
    img_filtered = np.abs(img_filtered) 

    return img_filtered, magnitude_spectrum, fshift_filtered

# FFT
filtered_fft, magnitude_spectrum, filtered_spectrum = fft_filter(corrupted)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("FFT Magnitude Spectrum")
plt.show()
def gamma_correction(image, gamma=1.2):
    image = np.clip(image, 0, 1)  # [0,1]
    return image ** gamma  

gamma_value = 0.8  
filtered_fft = gamma_correction(filtered_fft, gamma=gamma_value)
#Transformer
transformer = img_as_float(io.imread('CM.jpg', as_gray=True))  #from transformer
transformer = transformer.astype(np.float32)

# PSNR and SSIM
def calculate_metrics(img1, img2, method_name):
    psnr_value = metrics.peak_signal_noise_ratio(img1, img2, data_range=255)
    ssim_value = metrics.structural_similarity(img1, img2, data_range=255)
    print(f'{method_name} PSNR: {psnr_value:.6f} dB')
    print(f'{method_name} SSIM: {ssim_value:.6f}')

calculate_metrics(original, filtered_fft, "FFT Filter")
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

im3 = axes[0, 2].imshow(filtered_fft, cmap='gray')
axes[0, 2].set_title('c) Restored (FFT Filter)')
add_colorbar(im3, axes[0, 2])


im5 = axes[0, 3].imshow(transformer, cmap='gray')
axes[0, 3].set_title('e) Restored (Transformer)')
add_colorbar(im5, axes[0, 3])


axes[1, 0].hist(original.ravel(), bins=256, color='black',log=True)
axes[1, 0].set_title('f) Original Image Histogram')
axes[1, 0].grid(False)

axes[1, 1].hist(corrupted.ravel(), bins=256, color='black',log=True)
axes[1, 1].set_title('g) Corrupted Image Histogram')
axes[1, 1].grid(False)

axes[1, 2].hist(filtered_fft.ravel(), bins=256, color='black',log=True)
axes[1, 2].set_title('h) Restored (FFT Filter)')
axes[1, 2].grid(False)

axes[1, 3].hist(transformer.ravel(), bins=256, color='black',log=True)
axes[1, 3].set_title('j) Restored (Transformer)')
axes[1, 3].grid(False)

for i in range(5):  
    axes[0, i].axis('off')

plt.tight_layout()
plt.show()