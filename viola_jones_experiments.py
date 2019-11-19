import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib as plt
from pprint import pprint


def draw(tru, det, image):
    for i in range(len(tru)):
        cv.rectangle(image, (tru[i, 0], tru[i, 1]),
                     (tru[i, 2], tru[i, 3]), (0, 255, 0), 2)

    for i in range(len(det)):
        cv.rectangle(image, (det[i, 0], det[i, 1]),
                     (det[i, 2], det[i, 3]), (0, 0, 255), 2)


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

    try:
        TPR = successes/len(tru)
    except:
        TPR = 0
    try:
        PPV = successes/len(det)
    except:
        PPV = 0

    if (TPR == 0 or PPV == 0):
        F1 = 0.0
    else:
        F1 = 2*((TPR*PPV)/(TPR+PPV))

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


def face_detect(image_name, threshold):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[4:-4])

    det_faces = cv.CascadeClassifier(
        "frontalface.xml").detectMultiScale(image_grey, 1.1, 10)
    det_faces = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det_faces])

    tru_faces = get_faces(image_number)
    tru_faces = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru_faces])

    draw(tru_faces, det_faces, image)
    evaluate(tru_faces, det_faces, threshold)
    # cv.imshow("Faces", image)
    cv.imwrite("output_face_dart"+str(image_number)+".jpg", image)
    # cv.waitKey(0)


def get_faces(image_number):
    img4_faces = np.array([[354, 125, 114, 126]])
    img5_faces = np.array([[71, 150, 51, 53], [50, 250, 60, 65], [191, 221, 56, 58], [254, 173, 50, 50], [300, 246, 49, 62], [
                          381, 189, 60, 55], [428, 243, 56, 55], [512, 186, 52, 55], [554, 248, 62, 62], [645, 179, 50, 65], [677, 251, 54, 60]])
    img13_faces = np.array([[425, 120, 97, 135]])
    img14_faces = np.array([[471, 227, 76, 90], [731, 198, 94, 93]])
    img15_faces = np.array(
        [[56, 137, 70, 78], [365, 107, 86, 93], [534, 129, 84, 86]])

    if (image_number == 4):
        return img4_faces
    if (image_number == 5):
        return img5_faces
    if (image_number == 13):
        return img13_faces
    if (image_number == 14):
        return img14_faces
    if (image_number == 15):
        return img15_faces


def dart_detect(image_name, threshold):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[4:-4])

    det_darts = cv.CascadeClassifier(
        "dartboard.xml").detectMultiScale(image_grey, 1.1, 10)
    det_darts = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det_darts])

    tru_darts = get_darts(image_number)
    tru_darts = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru_darts])

    draw(tru_darts, det_darts, image)
    evaluate(tru_darts, det_darts, threshold)
    # cv.imshow("Darts", image)
    # cv.waitKey(0)
    cv.imwrite("output_dart_dart"+str(image_number)+".jpg", image)


def get_darts(image_number):
    img4_darts = np.array([[155, 64, 263, 263]])
    img5_darts = np.array([[416, 125, 129, 139]])

    if (image_number == 4):
        return img4_darts
    if (image_number == 5):
        return img5_darts


image_name = "dart4.jpg"
face_detect(image_name, 0.6)

image_name = "dart5.jpg"
face_detect(image_name, 0.6)

image_name = "dart13.jpg"
face_detect(image_name, 0.6)

image_name = "dart14.jpg"
face_detect(image_name, 0.6)

image_name = "dart15.jpg"
face_detect(image_name, 0.6)
# dart_detect(image_name, 0.6)
