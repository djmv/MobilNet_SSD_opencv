#Import the neccesary libraries
import numpy as np
import argparse
import cv2 

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network')
parser.add_argument("--image", default= "img.jpeg", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model'
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model'
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
# Load image fro
frame = cv2.imread(args.image)
frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
heightFactor = frame.shape[0]/300.0
widthFactor = frame.shape[1]/300.0 
# MobileNet requires fixed dimensions for input image(s)
# so we have to ensure that it is resized to 300x300 pixels.
# set a scale factor to image because network the objects has differents size. 
# We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 300, 300)
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
#Set to network the input blob 
net.setInput(blob)
#Prediction of network
detections = net.forward()

frame_copy = frame.copy()
frame_copy2 = frame.copy()
#Size of frame resize (300x300)
cols = frame_resized.shape[1] 
rows = frame_resized.shape[0]

#For get the class and location of object detected, 
# There is a fix index for class, location and confidence
# value in @detections array .
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > args.thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        # Draw location of object  
        cv2.rectangle(frame_resized, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0))

        cv2.rectangle(frame_copy, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                      (0, 255, 0),-1)
opacity = 0.3
cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > args.thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
          (0, 0, 0),2)
        # Draw label and confidence of prediction in frame resized
        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)

            yLeftBottom_ = max(yLeftBottom_, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_ - labelSize[1]),
                                 (xLeftBottom_ + labelSize[0], yLeftBottom_ + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom_, yLeftBottom_),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
            print label #print class and confidence 
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()