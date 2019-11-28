# encoding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the rectangle location of a face
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Face alignment
def face_alignment(faces):
    # face landmark detection predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth in 68 landmark points
        order = [36, 45, 30, 48, 54]
        # illustrate the 5 key facial landmark points
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(face, (x, y), 2, (0, 0, 255), -1)
        # compute the center of two eyes
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2,
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        # note: right - right
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # compute angle
        angle = math.atan2(dy,dx) * 180. / math.pi
        # compute affine matrix
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # perform a radiation transformation, ie rotation
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned

def alignment(imgpath, addsize):
    # pre-process
    im_raw = cv2.imread(imgpath).astype('uint8')
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    # process
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        detect_face = im_raw[(y - addsize) : (y + h + addsize), (x - addsize) : (x + w + addsize)]
        src_faces.append(detect_face)
        # illustrate the rectangle and text
        cv2.rectangle(im_raw, (x - addsize, y - addsize), (x + w + addsize, y + h + addsize), (0, 255, 0), 2)
        cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # get the aligned face
    faces_aligned = face_alignment(src_faces)
    # show the results
    cv2.imshow("src", im_raw)
    i = 0
    for face in faces_aligned:
        cv2.imshow("det_{}".format(i), face)
        i = i + 1
    cv2.waitKey(0)

if __name__ == "__main__":
    
    imgpath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\face\\target_family\\taiwan family\\12.jpg'
    addsize = 10
    alignment(imgpath, addsize)
    