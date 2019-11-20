import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib as plt
from pprint import pprint

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

def get_objects(image_type, image_number):
    img4_faces = np.array([[354, 125, 114, 126]])
    img5_faces = np.array([[71, 150, 51, 53], [50, 250, 60, 65], [191, 221, 56, 58], [254, 173, 50, 50], [300, 246, 49, 62], [381, 189, 60, 55], [428, 243, 56, 55], [512, 186, 52, 55], [554, 248, 62, 62], [645, 179, 50, 65], [677, 251, 54, 60]])
    img13_faces = np.array([[425, 120, 97, 135]])
    img14_faces = np.array([[471, 227, 76, 90], [731, 198, 94, 93]])
    img15_faces = np.array([[56, 137, 70, 78], [365, 107, 86, 93], [534, 129, 84, 86]])

    img4_darts = np.array([[155, 64, 263, 263]])
    img5_darts = np.array([[416, 125, 129, 139]])

    if (image_type == "faces"):
        if (image_number == 4): return img4_faces
        if (image_number == 5): return img5_faces
        if (image_number == 13): return img13_faces
        if (image_number == 14): return img14_faces
        if (image_number == 15): return img15_faces
    if (image_type == "darts"):
        if (image_number == 4): return img4_darts
        if (image_number == 5): return img5_darts

def viola_jones(image_name, image_type, threshold):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[5:-4])

    det = cv.CascadeClassifier(image_type+".xml").detectMultiScale(image_grey, 1.1, 3)
    det = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det])
    tru = get_objects(image_type, image_number)
    tru = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru])
    evaluate(tru, det, threshold)

    draw(image, det, (0, 255, 0), 2)
    draw(image, tru, (0, 0, 255), 2)
    cv.imwrite("viola_jones"+image_type+str(image_number)+".jpg", image)

viola_jones("darts4.jpg", "faces", 0.6)
viola_jones("darts5.jpg", "faces", 0.6)
viola_jones("darts13.jpg", "faces", 0.6)
viola_jones("darts14.jpg", "faces", 0.6)
viola_jones("darts15.jpg", "faces", 0.6)

viola_jones("darts4.jpg", "darts", 0.6)
viola_jones("darts5.jpg", "darts", 0.6)