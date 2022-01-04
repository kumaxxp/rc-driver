#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv

from RCdepth import RCdepth
from RCsegmentation import RCsegmentation

parser = argparse.ArgumentParser()
parser.add_argument("--movie", type=str, default=None)

args = parser.parse_args()

# Initialize video capture
print("open video file", args.movie)
cap = cv.VideoCapture(args.movie)
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv.CAP_PROP_FPS))
print(cap.get(cv.CAP_PROP_FRAME_COUNT))

if cap.isOpened() == False:
    print("file open ERROR!!")

rc_depth = RCdepth.RCdepth()
rc_segmentation = RCsegmentation.RCsegmentation()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Capture end")
        break

    image = copy.deepcopy(frame)

    segment_image, elapsed_time_seg = rc_segmentation.loop(image)    
    input_image = cv.resize(image, dsize = rc_segmentation.image_size)
    depth_image, elapsed_time = rc_depth.loop(input_image)

    # デバッグ用のイメージ保存
#    cv.imwrite('test.bmp', rc_segmentation.segmentation_map)    

    cv.imshow('depth', depth_image)
    cv.imshow('mask', rc_segmentation.segmentation_map)
    cv.imshow('segmentation', segment_image)
    
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()

