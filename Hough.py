import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib as plt
from pprint import pprint

def threshold(a, t):
    b = np.copy(a)
    b[b > t] = 255
    b[b <= t] = 0
    return b

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
    magnitude_image = np.sqrt(np.square(y_deriv_image)+np.square(x_deriv_image))
    direction_image = np.arctan2(y_deriv_image, x_deriv_image)
    direction_image[direction_image > np.pi/2] -= np.pi
    direction_image[direction_image < -np.pi/2] += np.pi
    return magnitude_image, direction_image

def hough(image_name, image_type, rmin, rmax, rinc, ainc, t1, t2, t3):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_number = int(image_name[5:-4])
    height, width = image.shape[:2]
    
    rmin = int(np.floor((rmin/100)*(np.min(np.array([height, width]))//2)))
    rmax = int(np.floor((rmax/100)*(np.min(np.array([height, width]))//2)))
    lmax = int(np.floor((np.sqrt(np.square(height//2)+np.square(width//2)))))

    circle_space = np.zeros((rmax, height, width))
    circle_image = np.zeros((height, width))
    line_space = np.zeros((lmax, 360))
    line_image = np.zeros((height, width))
    
    magnitude_image, direction_image = sobel(image_grey)
    for y in range(height):
        for x in range(width):
            if (image_type == "circles"):
                for r in range(rmin, rmax, rinc): 
                    if magnitude_image[y, x] > t1:
                        yp = y+(r*np.sin(direction_image[y, x]))
                        xp = x+(r*np.cos(direction_image[y, x]))
                        yn = y-(r*np.sin(direction_image[y, x]))
                        xn = x-(r*np.cos(direction_image[y, x]))
                        
                        if (yp > 0 and yp < height and xp > 0 and xp < width):
                            circle_space[r, int(yp), int(xp)] += 1
                        if (yn > 0 and yn < height and xn > 0 and xn < width):
                            circle_space[r, int(yn), int(xn)] += 1
            elif (image_type == "lines"):
                amin = int(np.clip(np.floor(np.degrees(direction_image[y, x])-6), 0, 360))
                amax = int(np.clip(np.floor(np.degrees(direction_image[y, x])+6+1), 0, 360))
                if ((amin > 10 and amax < 80) or (amin > 100 and amax < 170) or (amin > 190 and amax < 260) or (amin > 280 and amax < 350)):
                    for a in range(amin, amax, ainc): 
                        if magnitude_image[y, x] > t1:
                            d = int(np.floor((x-width//2)*np.cos(np.radians(a))+(y-height//2)*np.sin(np.radians(a))))
                            line_space[d, a] += 1

    circle_space = threshold(circle_space, t2)
    circle_image = np.sum(circle_space, axis=0)
    circle_image = threshold(circle_image, 1)

    line_space_prime = threshold(line_space, t3)
    while (np.size(np.nonzero(line_space_prime)) > 150):
        t3 += 1
        line_space_prime = threshold(line_space, t3)
    line_space = line_space_prime
    dn, an = np.nonzero(line_space)
    for a1 in range(len(an)-1):
        for a2 in range(a1+1, len(an)):
            for d1 in range(len(dn)-1):
                for d2 in range(d1+1, len(dn)):
                    if (line_space[dn[d1], an[a1]] == 255 and line_space[dn[d2], an[a2]] == 255 and abs(an[a1] - an[a2]) > 5):
                        a = np.array([[np.cos(np.radians(an[a1])), np.sin(np.radians(an[a1]))], [np.cos(np.radians(an[a2])), np.sin(np.radians(an[a2]))]])
                        d = np.array([[dn[d1]], [dn[d2]]])
                        if (np.linalg.det(a) != 0):
                            x, y = np.linalg.solve(a, d)
                            x, y = int(np.floor(x+width//2)), int(np.floor(y+height//2))
                            if (y > 0 and y < height and x > 0 and x < width):
                                line_image[y, x] = 255

    if (image_type == "circles"):
        cv.imwrite("hough_"+str(image_type)+"_output"+str(image_number)+".jpg", circle_image)
    elif (image_type == "lines"):
        cv.imwrite("hough_"+str(image_type)+"_output"+str(image_number)+".jpg", line_image)

rmin = 15
rmax = 35
rinc = 2
ainc = 2
mag_thresh = 200
circle_thresh = 8
line_thresh = 5
image_name = "input"+input("Please enter image number: ")+".jpg"
image_type = input("Please enter image type: ")
hough(image_name, image_type, rmin, rmax, rinc, ainc, mag_thresh, circle_thresh, line_thresh) 
'''
for n in range(16):
    image_name = "input"+str(n)+".jpg"
    hough(image_name, "circles", rmin, rmax, rinc, ainc, mag_thresh, circle_thresh, line_thresh)
    hough(image_name, "lines", rmin, rmax, rinc, ainc, mag_thresh, circle_thresh, line_thresh)
'''