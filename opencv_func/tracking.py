#!/usr/bin/env python
# -*- coding: utf-8 -*-

# OpenCVを使ってのトラッキング
# https://data-analysis-stats.jp/%E7%94%BB%E5%83%8F%E8%A7%A3%E6%9E%90/opencv%E3%81%AE%E7%89%A9%E4%BD%93%E8%BF%BD%E8%B7%A1/
#

import cv2
 
file = "/mnt/c/movie/pexels-tom-fisk-5786587.mp4"

def drawBox(img, bbox):
    # Box drawing function
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 0), 3, 1)
    cv2.putText(img, "Tracking", (15, 70), font, 0.5, (0, 0, 255), 2)

if __name__ == '__main__':
    cap = cv2.VideoCapture(file)  # 0 for Camera

    # Create tracker
    tracker = cv2.TrackerMIL_create()
    success, img = cap.read()

    # Create bbox

    #bbox = cv2.selectROI("Tracking", img, False)

    bbox = [520,484,84,100]
    tracker.init(img, bbox)

    font = cv2.FONT_HERSHEY_SIMPLEX


    while True:
        timer = cv2.getTickCount()
        success, img = cap.read()
        success, bbox = tracker.update(img)

        if success:
            drawBox(img, bbox)
        else:
            cv2.putText(img, "Tracking Lost", (15, 70), font, 0.5, (0, 0, 255), 2)

        # Frame rate per second
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        cv2.putText(img, "fps" + str(int(fps)), (15, 30), font, 0.5, (255, 255, 255), 2)
        cv2.imshow("Tracking", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
