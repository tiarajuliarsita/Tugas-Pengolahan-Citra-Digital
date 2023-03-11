import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('image02.png', 0)

# Define  the filter parameters
d0 = 50   # Cut-off frequency
n = 2     # Order of Butterworth filter

# Ideal Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), d0, 255, -1)
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * mask
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Butterworth Lowpass Filter
butterworth_filter = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_filter[i,j] = 1 / (1 + (d/d0)**(2*n))
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * butterworth_filter
img_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Display the results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_filtered, cmap='gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_filtered_butterworth, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
