# Nama: Ahmad Bisyral Hafi
# NIM: 220401010010
# Kelas: IFD51
# Nama Mata Kuliah: Pengolahan Citra Digital


import imageio
import numpy as np
import matplotlib.pyplot as plt

image_path = "low_contrast_image.jpg"
low_contrast_image = imageio.imread(image_path, mode='L')

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[image.astype('uint8')]

enhanced_image = histogram_equalization(low_contrast_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Citra Asli")
plt.imshow(low_contrast_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Citra Setelah Histogram Equalization")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

def adjust_contrast(image, level):
    return np.clip((image - 128) * level + 128, 0, 255).astype('uint8')

contrast_adjusted_image = adjust_contrast(low_contrast_image, 1.5)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Citra Asli")
plt.imshow(low_contrast_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Kontras Level 1.5")
plt.imshow(contrast_adjusted_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Histogram Equalization")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()