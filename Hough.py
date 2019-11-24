import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib as plt
from pprint import pprint

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

def hough(image_name, rmin, rmax, rinc, t1, t2):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_number = int(image_name[5:-4])
    height, width = image.shape[:2]
    
    rmin = int(np.floor((rmin/100)*(np.min(np.array([height, width]))//2)))
    rmax = int(np.floor((rmax/100)*(np.min(np.array([height, width]))//2)))

    magnitude_image, direction_image = sobel(image_grey)
    space = np.zeros((rmax, height, width))
    for y in range(height):
        for x in range(width):
            for r in range(rmin, rmax, rinc): 
                if magnitude_image[y, x] > t1:
                    yp = y+(r*np.sin(direction_image[y, x]))
                    xp = x+(r*np.cos(direction_image[y, x]))
                    yn = y-(r*np.sin(direction_image[y, x]))
                    xn = x-(r*np.cos(direction_image[y, x]))
                    
                    if (yp > 0 and yp < height and xp > 0 and xp < width):
                        space[r, int(yp), int(xp)] += 1
                    if (yn > 0 and yn < height and xn > 0 and xn < width):
                        space[r, int(yn), int(xn)] += 1
    space = threshold(space, t2)
    image = np.sum(space, axis=0)
    image = threshold(image, 1)
    cv.imwrite("hough_output"+str(image_number)+".jpg", image)

image_name = input("Please enter image name: ")
hough(image_name, 15, 35, 1, 200, 10) 
'''
hough("input0.jpg", 15, 35, 1, 200, 10)
hough("input1.jpg", 15, 35, 1, 200, 10)
hough("input2.jpg", 15, 35, 1, 200, 10)
hough("input3.jpg", 15, 35, 1, 200, 10)
hough("input4.jpg", 15, 35, 1, 200, 10)
hough("input5.jpg", 15, 35, 1, 200, 10)
hough("input6.jpg", 15, 35, 1, 200, 10)
hough("input7.jpg", 15, 35, 1, 200, 10)
hough("input8.jpg", 15, 35, 1, 200, 10)
hough("input9.jpg", 15, 35, 1, 200, 10)
hough("input10.jpg", 15, 35, 1, 200, 10)
hough("input11.jpg", 15, 35, 1, 200, 10)
hough("input12.jpg", 15, 35, 1, 200, 10)
hough("input13.jpg", 15, 35, 1, 200, 10)
hough("input14.jpg", 15, 35, 1, 200, 10)
hough("input15.jpg", 15, 35, 1, 200, 10)
'''