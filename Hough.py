import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib as plt

def threshold(a, t):
    a[a > t] = 255
    a[a <= t] = 0
    return a

def convolve(image,	kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output = np.zeros([iH, iW], dtype="float32")
    for y in np.arange(pad,	iH + pad):
        for x in np.arange(pad,	iW + pad):
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
            k = (roi*kernel).sum()
            output[y-pad, x-pad] = k
    return output

def sobel(image): 
    y_deriv_image = convolve(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    x_deriv_image = convolve(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    magnitude_image, direction_image = cv.cartToPolar(x_deriv_image, y_deriv_image)
    return magnitude_image, direction_image

def hough(magnitude_image, direction_image, rmin, rmax, ts, th):
    height, width = magnitude_image.shape[:2]
    if rmax == 0: 
        rmax = np.min(np.array([height, width]))//2
    hough_space = np.zeros((rmax, height, width))
    for r in range(rmin, rmax): 
        for y in range(height):
            for x in range(width):
                if magnitude_image[y, x] > ts:
                    yp = y+(r*np.sin(direction_image[y, x]))
                    xp = x+(r*np.cos(direction_image[y, x]))
                    yn = y-(r*np.sin(direction_image[y, x]))
                    xn = x-(r*np.cos(direction_image[y, x]))
                    
                    if (yp > 0 and yp < height and xp > 0 and xp < width):
                        hough_space[r, int(yp), int(xp)] += 1
                    if (yn > 0 and yn < height and xn > 0 and xn < width):
                        hough_space[r, int(yn), int(xn)] += 1
    hough_space = threshold(hough_space, th)
    return hough_space

image = cv.imread("coins1.png")
image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
magnitude_image, direction_image = sobel(image)
hough_space = hough(magnitude_image, direction_image, 30, 60, 200, 8)
hough_image = np.sum(hough_space, axis=0)
hough_image = cv.normalize(hough_image, hough_image, 0, 1, cv.NORM_MINMAX)
cv.imshow("Hough", hough_image)
cv.waitKey(0)
