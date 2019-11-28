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
    img4_faces = np.array([[331, 103, 157, 165]])
    img5_faces = np.array([[59, 139, 65, 64], [51, 250, 65, 69], [192, 212, 57, 72], [242, 167, 66, 66], [289, 238, 56, 74], [375, 192, 68, 58], [426, 232, 58, 71], [514, 176, 65, 66], [555, 240, 69, 76], [646, 186, 63, 64], [681, 244, 52, 68]])
    img13_faces = np.array([[420, 131, 118, 128]])
    img14_faces = np.array([[456, 215, 99, 108], [727, 190, 104, 105]])
    img15_faces = np.array([[67, 133, 56, 80], [541, 136, 52, 76]])

    if (image_type == "darts"):
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
    if (image_type == "faces"):
        if (image_number == 4): return img4_faces
        if (image_number == 5): return img5_faces
        if (image_number == 13): return img13_faces
        if (image_number == 14): return img14_faces
        if (image_number == 15): return img15_faces

def viola_jones(image_name, image_type, nmin, threshold):
    image = cv.imread(image_name)
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_grey = cv.equalizeHist(image_grey)
    image_number = int(image_name[5:-4])

    det = cv.CascadeClassifier(image_type+".xml").detectMultiScale(image_grey, 1.1, nmin)
    det = np.array([[x, y, w+x, h+y] for (x, y, w, h) in det])
    tru = get_objects(image_type, image_number)
    tru = np.array([[x, y, w+x, h+y] for (x, y, w, h) in tru])
    evaluate(tru, det, threshold)

    draw(image, det, (0, 255, 0), 2)
    draw(image, tru, (0, 0, 255), 2)
    cv.imwrite("viola_jones_"+image_type+"_output"+str(image_number)+".jpg", image)
 
nmin = 8
iou_thresh = 0.3
image_name = "input"+input("Please enter image number: ")+".jpg"
image_type = input("Please enter image type: ")
viola_jones(image_name, image_type, nmin, iou_thresh)
'''
for n in range(16):
    image_name = "input"+str(n)+".jpg"
    viola_jones(image_name, "darts", nmin, iou_thresh)

i = np.array([4, 5, 13, 14, 15])
for n in i:
    image_name = "input"+str(n)+".jpg"
    viola_jones(image_name, "faces", nmin, iou_thresh)
'''