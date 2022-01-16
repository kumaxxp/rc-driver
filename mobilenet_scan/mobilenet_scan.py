import argparse
import cv2
from cv2 import dnn
import numpy as np
import time

inWidth = 224
inHeight = 224
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.017
meanVal = (103.94, 116.78, 123.68)
prevFrameTime = None
currentFrameTime = None

# https://github.com/shicai/MobileNet-Caffe から、モジュールや設定フィルを得る

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="number of video device", default=0)
    parser.add_argument("--prototxt", default="mobilenet_v2_deploy.prototxt")
    parser.add_argument("--caffemodel", default="mobilenet_v2.caffemodel")
    parser.add_argument("--classNames", default="synset.txt")
    parser.add_argument("--preview", default=True)
    parser.add_argument("--movie", type=str, default=None)

    args = parser.parse_args()
    net = dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
    cap = cv2.VideoCapture(args.movie)
    f = open(args.classNames, 'r')
    classNames = f.readlines()
    showPreview = (args.preview == True or args.preview == "True" or args.preview == "true")

    while True:
        ret, frame = cap.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = dnn.blobFromImage(rgbFrame, inScaleFactor, (inWidth, inHeight), meanVal)
        net.setInput(blob)
        detections = net.forward()

        maxClassId = 0
        maxClassPoint = 0
        for i in range(detections.shape[1]):
            classPoint = detections[0, i, 0, 0]
            if (classPoint > maxClassPoint):
                maxClassId = i
                maxClassPoint = classPoint

        className = classNames[maxClassId]
        print("class id: ", maxClassId)
        print("class point: ", maxClassPoint)
        print("name: ", className)
        prevFrameTime = currentFrameTime
        currentFrameTime = time.time()
        if (prevFrameTime != None):
            print(1.0 / (currentFrameTime - prevFrameTime), "fps")

        if (showPreview):
            font = cv2.FONT_HERSHEY_SIMPLEX
            size = 1
            color = (255,255,255)
            weight = 2
            cv2.putText(frame, className, (10, 30), font, size, color, weight)
            cv2.putText(frame, str(maxClassPoint), (10, 60), font, size, color, weight)
            cv2.imshow("detections", frame)

        if cv2.waitKey(1) >= 0:
            pass