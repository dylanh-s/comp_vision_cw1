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

def hough(image_name, rmin, rmax, ts, th):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    height, width = image.shape[:2]
    magnitude_image, direction_image = sobel(image_grey)
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

def draw(tru, det, image):
    for i in range(len(tru)):
        cv.rectangle(image, (tru[i, 0], tru[i, 1]), (tru[i, 2], tru[i, 3]), (0, 255, 0), 2)

    for i in range(len(det)):
        cv.rectangle(image, (det[i, 0], det[i, 1]), (det[i, 2], det[i, 3]), (0, 0, 255), 2)

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

def viola_jones(image_name, image_type, threshold):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[4:-4])

    det_faces = cv.CascadeClassifier(image_type+".xml").detectMultiScale(image_grey, 1.1, 3)
    det_faces = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det_faces])

    tru_faces = get_objects(image_type, image_number)
    tru_faces = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru_faces])

    draw(tru_faces, det_faces, image)
    evaluate(tru_faces, det_faces, threshold)
    #cv.imwrite("output"+image_type+str(image_number)+".jpg", image)
    #cv.imshow(image_type, image)
    #cv.waitKey(0)
    return image

def get_objects(image_type, image_number):
    img4_faces = np.array([[354, 125, 114, 126]])
    img5_faces = np.array([[71, 150, 51, 53], [50, 250, 60, 65], [191, 221, 56, 58], [254, 173, 50, 50], [300, 246, 49, 62], [381, 189, 60, 55], [428, 243, 56, 55], [512, 186, 52, 55], [554, 248, 62, 62], [645, 179, 50, 65], [677, 251, 54, 60]])
    img13_faces = np.array([[425, 120, 97, 135]])
    img14_faces = np.array([[471, 227, 76, 90], [731, 198, 94, 93]])
    img15_faces = np.array([[56, 137, 70, 78], [365, 107, 86, 93], [534, 129, 84, 86]])

    img4_darts = np.array([[155, 64, 263, 263]])
    img5_darts = np.array([[416, 125, 129, 139]])
    img14_darts = np.array([[123, 103, 123, 123], [989, 97, 121, 121]])

    if (image_type == "faces"):
        if (image_number == 4): return img4_faces
        if (image_number == 5): return img5_faces
        if (image_number == 13): return img13_faces
        if (image_number == 14): return img14_faces
        if (image_number == 15): return img15_faces
    if (image_type == "darts"):
        if (image_number == 4): return img4_darts
        if (image_number == 5): return img5_darts
        if (image_number == 14): return img14_darts

def viola_hough(image_name):
    viola_image = viola_jones(image_name, "darts", 0.6)
    hough_space = hough(image_name, 10, 70, 200, 8)
    hough_image = np.sum(hough_space, axis=0)
    cv.imshow("Viola", viola_image)
    cv.imshow("Hough", hough_image)
    cv.waitKey(0)



image_name = "dart4.jpg"
#viola_jones(image_name, "faces", 0.6)
#viola_jones(image_name, "darts", 0.6)

image_name = "dart5.jpg"
#viola_jones(image_name, "faces", 0.6)
#viola_jones(image_name, "darts", 0.6)

image_name = "dart13.jpg"
#viola_jones(image_name, "faces", 0.6)
#viola_jones(image_name, "darts", 0.6)

image_name = "dart14.jpg"
#viola_jones(image_name, "faces", 0.6)
viola_hough(image_name)



image_name = "dart15.jpg"
#viola_jones(image_name, "faces", 0.6)
#viola_jones(image_name, "darts", 0.6)
