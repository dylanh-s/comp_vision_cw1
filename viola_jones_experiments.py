import cv2
import numpy as np
import matplotlib as plt
import scipy as sc
from pprint import pprint
from shapely.geometry import Polygon

THRESHOLD = 0.6
successful_detections = 0


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def face_detect(image, image_no, THRESHOLD=0.55):
    successful_detections = 0
    cascade = cv2.CascadeClassifier('frontalface.xml')
    frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(frame_gray, frame_gray)
    faces = cascade.detectMultiScale(frame_gray, 1.1, 10)
    (no_det_faces, m) = faces.shape[:2]
    detected_faces = np.ones((no_det_faces, 4))
    # pprint(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = frame_gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

    for i in range(no_det_faces):
        (x, y, w, h) = faces[i]
        detected_faces[i][:] = [x, y, w+x, h+y]

    # add ground truth rectangles
    face_pos, face_size = get_face_positions(image_no)
    (no_faces, j) = face_pos.shape[:2]
    true_faces = np.ones((no_faces, 4))
    for i in range(no_faces):
        (w, h) = face_size[i][:]
        #pprint((w, h))
        (x, y) = face_pos[i][:]
        #pprint((x, y))
        true_faces[i][:] = [x, y, w+x, h+y]
        pprint(true_faces)
        cv2.rectangle(image, (x, y),
                      (x+w, y+h), (0, 255/(i+1), 0), 2)
    cv2.imshow('lol', image)

    for t_f in true_faces:
        iou_max = 0
        for d_f in detected_faces:
            iou_score = bb_intersection_over_union(d_f, t_f)
            if (iou_score >= iou_max):
                iou_max = iou_score
        if (iou_max > THRESHOLD):
            successful_detections += 1
        pprint(iou_max)
    false_detections = 0
    # for d_f in detected_faces:
    #     iou_max = 0
    #     for t_f in true_faces:
    #         iou_score = bb_intersection_over_union(d_f, t_f)
    #         if (iou_score >= iou_max):
    #             iou_max = iou_score
    #     if (iou_max < THRESHOLD):
    #         false_detections += 1

    TPR = successful_detections/no_faces
    #FNR = false_detections/no_det_faces
    PPV = successful_detections / no_det_faces
    F1 = 2*((TPR*PPV)/(TPR+PPV))
    pprint("TPR = " + str(TPR))
    pprint("PPV = " + str(PPV))
    pprint("F1 = " + str(F1))
    # successful_detectionss
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_face_positions(image_number=4):
    img4_face_positions = np.array([[354, 125]])
    img4_face_sizes = np.array([[114, 126]])

    img5_face_positions = np.array([[71, 150], [50, 250], [191, 221], [254, 173], [
        300, 246], [381, 189], [428, 243], [512, 186], [554, 248], [645, 179], [677, 251]])
    img5_face_sizes = np.array([[51, 53], [60, 65], [56, 58], [50, 50], [49, 62], [
        60, 55], [56, 55], [52, 55], [62, 62], [50, 65], [54, 60]])

    img13_face_positions = np.array([[425, 120]])
    img13_face_sizes = np.array([[97, 135]])

    img14_face_positions = np.array([[471, 227], [731, 198]])
    img14_face_sizes = np.array([[76, 90], [94, 93]])

    img15_face_positions = np.array([[56, 137], [365, 107], [534, 129]])
    img15_face_sizes = np.array([[70, 78], [86, 93], [84, 86]])

    if (image_number == 4):
        return img4_face_positions, img4_face_sizes
    if (image_number == 5):
        return img5_face_positions, img5_face_sizes
    if (image_number == 13):
        return img13_face_positions, img13_face_sizes
    if (image_number == 14):
        return img14_face_positions, img14_face_sizes
    if (image_number == 15):
        return img15_face_positions, img15_face_sizes


image_no = 5
image = cv2.imread("dart5.jpg")
face_detect(image, image_no)
