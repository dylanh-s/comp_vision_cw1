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
    magnitude_image = np.sqrt(np.square(y_deriv_image)+np.square(x_deriv_image))
    direction_image = np.arctan2(y_deriv_image, x_deriv_image)
    direction_image[direction_image > np.pi/2] -= np.pi
    direction_image[direction_image < -np.pi/2] += np.pi
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
    det = []
    for y in range(height):
        for x in range(width):
            if (image[y, x] == 255):
                det.append([y, x])
    det = np.asarray(det)
    #cv.imshow("Hough", image)
    #cv.waitKey(0)
    return det

def draw(image, boxes, colour, width):
    for i in range(len(boxes)):
        cv.rectangle(image, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), colour, width)

def intersection_over_union(tru, det):
    x1 = max(tru[0], det[0])
    y1 = max(tru[1], det[1])
    x2 = min(tru[2], det[2])
    y2 = min(tru[3], det[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    tru_area = (tru[2] - tru[0] + 1) * (tru[3] - tru[1] + 1)
    det_area = (det[2] - det[0] + 1) * (det[3] - det[1] + 1)
    union_area = tru_area + det_area - intersection_area
    return intersection_area/union_area

def evaluate(tru, det, threshold):
    successes = 0
    for t in tru:
        iou_max = 0
        for d in det:
            iou_score = intersection_over_union(t, d)
            if (iou_score >= iou_max):
                iou_max = iou_score
        if (iou_max > threshold):
            successes += 1

    try: TPR = successes/len(tru)
    except: TPR = 0.0
    try: PPV = successes/len(det)
    except: PPV = 0.0
    try: F1 = 2*((TPR*PPV)/(TPR+PPV))
    except: F1 = 0.0

    print("TPR = " + str(TPR))
    print("PPV = " + str(PPV))
    print("F1  = " + str(F1))

def get_objects(image_number):
    img0_darts = np.array([[442, 16, 153, 175]])
    img1_darts = np.array([[198, 132, 192, 191]])
    img2_darts = np.array([[104, 98, 85, 86]])
    img3_darts = np.array([[326, 150, 62, 68]])
    img4_darts = np.array([[186, 96, 205, 205]])
    img5_darts = np.array([[434, 142, 103, 107]])
    img6_darts = np.array([[213, 118, 59, 60]])
    img7_darts = np.array([[257, 172, 142, 142]])
    img8_darts = np.array([[68, 253, 57, 86], [845, 219, 112, 117]])
    img9_darts = np.array([[205, 47, 229, 231]])
    img10_darts = np.array([[93, 106, 92, 106], [586, 129, 53, 82], [917, 150, 33, 63]])
    img11_darts = np.array([[177, 106, 54, 79]])
    img12_darts = np.array([[158, 79, 56, 133]])
    img13_darts = np.array([[275, 122, 127, 127]])
    img14_darts = np.array([[123, 103, 121, 121], [989, 98, 120, 120]])
    img15_darts = np.array([[155, 57, 128, 137]])

    if (image_number == 0): return img0_darts
    if (image_number == 1): return img1_darts
    if (image_number == 2): return img2_darts
    if (image_number == 3): return img3_darts
    if (image_number == 4): return img4_darts
    if (image_number == 5): return img5_darts
    if (image_number == 6): return img6_darts
    if (image_number == 7): return img7_darts
    if (image_number == 8): return img8_darts
    if (image_number == 9): return img9_darts
    if (image_number == 10): return img10_darts
    if (image_number == 11): return img11_darts
    if (image_number == 12): return img12_darts
    if (image_number == 13): return img13_darts
    if (image_number == 14): return img14_darts
    if (image_number == 15): return img15_darts

def viola_jones(image_name, t3):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[5:-4])

    det = cv.CascadeClassifier("darts.xml").detectMultiScale(image_grey, 1.1, 3)
    det = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det])
    tru = get_objects(image_number)
    tru = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru])
    evaluate(tru, det, t3)

    draw(image, det, (0, 0, 255), 2)
    draw(image, tru, (0, 255, 0), 2)
    #cv.imshow("Viola", image)
    #cv.waitKey(0)
    return det, tru

def viola_hough(image_name, rmin, rmax, rinc, t1, t2, t3, t4): 
    #t1 & t2 are hough thresholds, t3 is viola_jones threshold, t4 is viola_hough threshold
    image = cv.imread(image_name)
    image_number = int(image_name[5:-4])
    height, width = image.shape[:2]
    det_circles = hough(image_name, rmin, rmax, rinc, t1, t2)
    det_objects, tru_objects = viola_jones(image_name, t3)
    
    det_combo = []
    for (x1, y1, x2, y2) in det_objects:
        votes = 0
        for (y, x) in det_circles:
            if (y >= y1 and y <= y2 and x >= x1 and x <= x2):
                votes += 1
        print(votes)
        if (votes > t4):
            det_combo.append([x1, y1, x2, y2])
    det_combo = np.asarray(det_combo)
    
    pprint(det_circles)
    pprint(det_objects)
    pprint(det_combo)
    draw(image, det_combo, (0, 255,  0), 2)
    draw(image, tru_objects, (0, 0, 255), 2)
    cv.imwrite("viola_hough_output"+str(image_number)+".jpg", image)
    cv.imshow("Combo", image)
    cv.waitKey(0)

image_name = input("Please enter image name: ")
viola_hough(image_name, 15, 35, 1, 200, 10, 0.6, 40)