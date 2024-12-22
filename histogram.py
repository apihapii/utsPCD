import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from PIL import Image

image_path = 'C:/Users/mafth/Downloads/tugas 1/rgb2.png'  # Sesuaikan path gambar
rgb_image = iio.imread(image_path)
print(f"Shape of loaded image: {rgb_image.shape}")


if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:

    grayscale_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
elif len(rgb_image.shape) == 2:

    grayscale_image = rgb_image
else:
    raise ValueError("Unsupported image format. Please use RGB or grayscale images.")

print(f"Shape of grayscale image: {grayscale_image.shape}")

grayscale_output_path = 'C:/Users/mafth/Downloads/tugas 1/grayscale_image.png'
Image.fromarray(grayscale_image).save(grayscale_output_path)
print(f"Grayscale image saved to: {grayscale_output_path}")

histogram, bin_edges = np.histogram(grayscale_image, bins=256, range=(0, 255))

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], histogram, width=1, edgecolor="black", color="gray")
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)
plt.grid(axis='y', linestyle='--', alpha=0.7)

histogram_output_path = 'C:/Users/mafth/Downloads/tugas 1/histogram.png'
plt.savefig(histogram_output_path)
plt.show()
print(f"Histogram saved to: {histogram_output_path}")

dominant_intensity = np.argmax(histogram)
print(f"Dominant intensity: {dominant_intensity} with {histogram[dominant_intensity]} pixels")
