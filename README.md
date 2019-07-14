# Animal-Face-Detectors

The face detectors for human, cat, dog using dlib and opencv

## Reference

All the APIs are provided by opencv, and the document is [here](https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale).

## CascadeClassifier::detectMultiScale

Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

- C++: void CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
  
- Python: cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) → objects

- Python: cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) → objects

- C: CvSeq* cvHaarDetectObjects(const CvArr* image, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scale_factor=1.1, int min_neighbors=3, int flags=0, CvSize min_size=cvSize(0,0), CvSize max_size=cvSize(0,0) )

- Python: cv.HaarDetectObjects(image, cascade, storage, scale_factor=1.1, min_neighbors=3, flags=0, min_size=(0, 0)) → detectedObjects

- Parameters:

1. cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).

2. image – Matrix of the type CV_8U containing an image where objects are detected.

3. objects – Vector of rectangles where each rectangle contains the detected object.

4. scaleFactor – Parameter specifying how much the image size is reduced at each image scale.

5. minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.

6. flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.

7. minSize – Minimum possible object size. Objects smaller than that are ignored.

8. maxSize – Maximum possible object size. Objects larger than that are ignored.
