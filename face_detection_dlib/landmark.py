# encoding:utf-8

import dlib
import numpy as np
import cv2

# Get the rectangle location of a face
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Convert the 68 features into numpy ndarray format
def shape_to_np(shape, dtype = "int"):
    coords = np.zeros((68, 2), dtype = dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Resize the pending image
def resize(image, width = 1500):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def feature(imgpath):
    # pre-process
    image = cv2.imread(imgpath)
    image = resize(image, width = 1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # process
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shapes.append(shape)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for shape in shapes:
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    # show
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    
    imgpath = 'C:\\Users\\yzzha\\Desktop\\code\\face\\target_family\\taiwan family\\1.jpg'
    feature(imgpath)
    