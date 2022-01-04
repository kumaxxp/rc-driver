#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime

class RCdepth:
    def __init__(self) -> None:
        self.model_path='saved_model_lite_hr_depth_k_t_encoder_depth_384x1280/lite_hr_depth_k_t_encoder_depth_384x1280.onnx'
        self.input_size=[384, 1280]
#        self.model_path='saved_model_lite_hr_depth_k_t_encoder_depth_192x640/lite_hr_depth_k_t_encoder_depth_192x640.onnx'
#        self.input_size=[192, 640]
        print("model ", self.model_path)
        self.onnx_session = onnxruntime.InferenceSession(self.model_path)

    def run_inference(self, onnx_session, input_size, image):
        # Pre process:Resize, BGR->RGB, Transpose, float32 cast
        input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0

        # Inference
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        result = onnx_session.run([output_name], {input_name: input_image})

        # Post process
        depth_map = result[0]
        d_min = np.min(depth_map)
        d_max = np.max(depth_map)
        depth_map = (depth_map - d_min) / (d_max - d_min)
        depth_map = depth_map * 255.0
        depth_map = np.asarray(depth_map, dtype="uint8")

        depth_map = depth_map.reshape(input_size[0], input_size[1])

        return depth_map

    def draw_debug(self, image_width, image_height, depth_map):
        # 深度画像を生成する
        depth_image = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
        depth_image = cv.resize(depth_image, dsize=(image_width, image_height))

        return depth_image

    def loop(self, frame):
        start_time = time.time()
        depth_map = self.run_inference(self.onnx_session, self.input_size, frame)
        elapsed_time = time.time() - start_time

        depth_image = self.draw_debug(frame.shape[1], frame.shape[0], depth_map)

        return depth_image, elapsed_time


if __name__ == '__main__':
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


    rc_depth = RCdepth()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture end")
            break

        image = copy.deepcopy(frame)

        depth_image, elapsed_time = rc_depth.loop(image)

        cv.imshow('road-segmentation-adas-0001 Demo', depth_image)
        
        cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()

