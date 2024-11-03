import numpy as np
import imageio.v2 as img
import matplotlib.pyplot as plt

image = img.imread("C:/Users/mafth/Downloads/tugas 1/source.jpg")

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

imgRed = np.zeros_like(image)
imgGreen = np.zeros_like(image)
imgBlue = np.zeros_like(image)

imgRed[:, :, 0] = red
imgGreen[:, :, 1] = green
imgBlue[:, :, 2] = blue

plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(4, 1, 2)
plt.title("Red Channel")
plt.imshow(imgRed)

plt.subplot(4, 1, 3)
plt.title("Green Channel")
plt.imshow(imgGreen)

plt.subplot(4, 1, 4)
plt.title("Blue Channel")
plt.imshow(imgBlue)

plt.show()
