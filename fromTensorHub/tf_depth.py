#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np

import urllib.request
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

class tf_depth:
    def __init__(self) -> None:
        self.module = hub.load("https://tfhub.dev/intel/midas/v2_1_small/1", tags=['serve'])
        self.model = self.module.signatures['serving_default']
        self.input_size = (256,256)


    def run_inference(self, input_size, image):
        # リサイズ
        resize_image = cv2.resize(image, (256, 256))

        # 正規化
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB) / 255.0

        # 形状変更
        resize_image = resize_image.transpose(2, 0, 1)
        resize_image = resize_image.reshape(1, 3, 256, 256)

        # tensor形式へ変換
        tensor = tf.convert_to_tensor(resize_image, dtype=tf.float32)

        result = self.model(tensor)

        predict_result = result['default'].numpy()
        predict_result = np.squeeze(predict_result)

        predict_result = cv2.resize(predict_result, (image.shape[1], image.shape[0]))

        # 最大値が255になるよう変換
        depth_max = predict_result.max()
        color_map = ((predict_result / depth_max) * 255).astype(np.uint8)

        # カラーマップ画像へ変換
        color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_TURBO)

        return color_map

    def loop(self, frame):
        start_time = time.time()
        depth_map = self.run_inference(self.input_size, frame)
        elapsed_time = time.time() - start_time

        return depth_map, elapsed_time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", type=str, default=None)

    args = parser.parse_args()

    # Initialize video capture
    print("open video file", args.movie)
    cap = cv2.VideoCapture(args.movie)

    if cap.isOpened() == False:
        print("file open ERROR!!")


    rc_depth = tf_depth()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture end")
            break

        image = copy.deepcopy(frame)

        #resize_image = cv2.resize(frame,dsize=(320,240))

        depth_image, elapsed_time = rc_depth.loop(image)
        print(elapsed_time)

        cv2.imshow('depth', depth_image)
        
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

