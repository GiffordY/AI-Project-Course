"""
@GiffordY
Some edge detection operators in OpenCV
"""

import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('lenna.png', 1)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel
img_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
img_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
img_x_abs = cv.convertScaleAbs(img_x)
img_y_abs = cv.convertScaleAbs(img_y)
img_sobel = cv.addWeighted(img_x_abs, 0.5, img_y_abs, 0.5, 0)

# Laplace
img_laplace = cv.Laplacian(img_gray, cv.CV_64F, ksize=3)

# Canny
img_canny = cv.Canny(img_gray, 120, 160)

# Show edge detection results
plt.subplot(231), plt.title("Gray Image"), plt.imshow(img_gray, cmap='gray')
plt.subplot(232), plt.title("Laplace"), plt.imshow(img_laplace, cmap='gray')
plt.subplot(233), plt.title("Canny"), plt.imshow(img_canny, cmap='gray')
plt.subplot(234), plt.title("Sobel X"), plt.imshow(img_x_abs, cmap='gray')
plt.subplot(235), plt.title("Sobel Y"), plt.imshow(img_y_abs, cmap='gray')
plt.subplot(236), plt.title("Sobel"), plt.imshow(img_sobel, cmap='gray')
plt.show()
