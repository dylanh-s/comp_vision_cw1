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
    magnitude_image, direction_image = cv.cartToPolar(x_deriv_image, y_deriv_image)
    return magnitude_image, direction_image

def hough(image_name, rmin, rmax, rinc, ainc, t1, t2, t3):
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

    det_circles = []
    det_lines = []
    for y in range(height):
        for x in range(width):
            if (circle_image[y, x] == 255):
                det_circles.append([y, x])
            if (line_image[y, x] == 255):
                det_lines.append([y, x])
    det_circles = np.asarray(det_circles)
    det_lines = np.asarray(det_lines)
    return det_circles, det_lines

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
    ''' 
    #Smaller ones!
    img0_darts = np.array([[471, 49, 93, 109]])
    img1_darts = np.array([[234, 168, 119, 119]])
    img2_darts = np.array([[120, 115, 53, 53]])
    img3_darts = np.array([[336, 162, 41, 43]])
    img4_darts = np.array([[225, 135, 127, 127]])
    img5_darts = np.array([[454, 162, 64, 66]])
    img6_darts = np.array([[224, 130, 37, 37]])
    img7_darts = np.array([[283, 199, 88, 88]])
    img8_darts = np.array([[79, 269, 36, 54], [866, 242, 68, 72]])
    img9_darts = np.array([[248, 92, 143, 143]])
    img10_darts = np.array([[111, 125, 57, 67], [596, 145, 33, 51], [924, 162, 21, 41]])
    img11_darts = np.array([[187, 121, 35, 44]])
    img12_darts = np.array([[169, 104, 35, 84]])
    img13_darts = np.array([[299, 146, 80, 80]])
    img14_darts = np.array([[145, 126, 76, 76], [1012, 120, 75, 75]])
    img15_darts = np.array([[182, 83, 79, 84]])
    '''
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

def viola_jones(image_name, nmin):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[5:-4])

    det = cv.CascadeClassifier("darts.xml").detectMultiScale(image_grey, 1.1, nmin)
    det = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det])
    tru = get_objects(image_number)
    tru = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru])

    draw(image, det, (0, 0, 255), 2)
    draw(image, tru, (0, 255, 0), 2)
    return det, tru

def viola_hough(image_name, rmin, rmax, rinc, ainc, nmin, t1, t2, t3, t4, t5, t6): 
    #t1, t2, t3 are hough thresholds, t4, t5, t6 are viola_hough thresholds
    image = cv.imread(image_name)
    image_number = int(image_name[5:-4])
    height, width = image.shape[:2]
    det_circles, det_lines = hough(image_name, rmin, rmax, rinc, ainc, t1, t2, t3)
    det_objects, tru_objects = viola_jones(image_name, nmin)
    
    det_combo = []
    for (x1, y1, x2, y2) in det_objects:
        circle_votes = 0
        line_votes = 0
        for (y, x) in det_circles:
            if (y >= y1 and y <= y2 and x >= x1 and x <= x2):
                circle_votes += 1
        for (y, x) in det_lines:
            if (y >= y1 and y <= y2 and x >= x1 and x <= x2):
                line_votes += 1
        if (circle_votes > t4 or line_votes > t5):
            det_combo.append([x1, y1, x2, y2])
    det_combo = np.asarray(det_combo)
    evaluate(tru_objects, det_combo, t6)
    
    draw(image, det_combo, (0, 255,  0), 2)
    draw(image, tru_objects, (0, 0, 255), 2)
    cv.imwrite("viola_hough_output"+str(image_number)+".jpg", image)

rmin = 15
rmax = 35
rinc = 2
ainc = 2
nmin = 8
mag_thresh = 200
circle_thresh = 8
line_thresh = 5
circle_votes = 5
line_votes = 20
iou_thresh = 0.3
image_name = "input"+input("Please enter image number: ")+".jpg"
viola_hough(image_name, rmin, rmax, rinc, ainc, nmin, mag_thresh, circle_thresh, line_thresh, circle_votes, line_votes, iou_thresh) 
'''
for n in range(16):
    image_name = "input"+str(n)+".jpg"
    viola_hough(image_name, rmin, rmax, rinc, ainc, nmin, mag_thresh, circle_thresh, line_thresh, circle_votes, line_votes, iou_thresh)
'''