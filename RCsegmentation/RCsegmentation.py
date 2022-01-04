#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

# 実行方法の例
# python3 demo/demo_road-segmentation-adas-0001_onnx.py --movie /mnt/c/movie/production\ ID_4608593.mp4
# どのサイズの動画でも適当なサイズに変換して、道路をセグメンテーションする。
#

import cv2 as cv
import numpy as np
import onnxruntime

class RCsegmentation:

    def __init__(self) -> None:
        self.model_path ='saved_model/model_float32.onnx'
        self.size ='512,896'
        self.input_size = [512,896]
        self.onnx_session = onnxruntime.InferenceSession(self.model_path)
        self.score = 0.5
        self.__image_size = (0, 0)
        self.__h_range = 0
        self.__segmentation_map = None

    @property
    def image_size(self):
        pass

    @image_size.getter
    def image_size(self):
        return self.__image_size

    @property
    def h_range(self):
        pass

    @h_range.getter
    def h_range(self):
        return self.__h_range

    @property
    def segmentation_map(self):
        pass

    @segmentation_map.getter
    def segmentation_map(self):
        return self.__segmentation_map


    def run_inference(self, input_size, image):
        # Pre process:Resize, expand dimensions, float32 cast
        input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        result = self.onnx_session.run([output_name], {input_name: input_image})

        # Post process:squeeze
        segmentation_map = result[0]
        segmentation_map = np.squeeze(segmentation_map)

        return segmentation_map


    def loop(self, image):

        start_time = time.time()

        # 動画からキャプチャした画像を縮小する
        h = self.input_size[0]
        w = self.input_size[1]
        size = (w,h)
        resize_image = cv.resize(image, size)

        # 縮小した画像のサイズ
        self.__image_size = size

        # セグメンテーション実行
        segmentation_map = self.run_inference(self.input_size, resize_image)
        self.__segmentation_map = copy.deepcopy(segmentation_map)

        self.elapsed_time = time.time() - start_time

        # セグメンテーションを元画像にオーバーレイ
        debug_image = self._draw_debug(resize_image, self.score, segmentation_map)
#        debug_image = self._draw_mask(resize_image, self.score, segmentation_map)

        

        # 物体検出実行
        # 元画像cv_bgrをsrc_ptsを台形に切り取って、鳥観図に変換する        
#        debug_image = self._BirdsEyeViewTransfer(debug_image, False)

        return debug_image, self.elapsed_time


    def _draw_debug(self, image, score, segmentation_map):
        image_width, image_height = image.shape[1], image.shape[0]

        # Match the size
        debug_image = copy.deepcopy(image)
        segmentation_map = cv.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv.INTER_LINEAR,
        )

        # color list
        color_image_list = []
        # ID 0:BackGround
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)
        color_image_list.append(bg_image)
        # ID 1:Road
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (255, 0, 0)
        color_image_list.append(bg_image)
        # ID 2:Curb
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)
        color_image_list.append(bg_image)
        # ID 4:Mark
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 255)
        color_image_list.append(bg_image)

        # Overlay segmentation map
        masks = segmentation_map.transpose(2, 0, 1)
        for index, mask in enumerate(masks):
            # Threshold check by score
            mask = np.where(mask > score, 0, 1)

            # Overlay
            mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
            mask_image = np.where(mask, debug_image, color_image_list[index])
            debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)
        
        return debug_image

    def _draw_mask(self, image, score, segmentation_map):
        image_width, image_height = image.shape[1], image.shape[0]

        # Match the size
        debug_image = copy.deepcopy(image)
        segmentation_map = cv.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv.INTER_LINEAR,
        )

        # color list
        color_image_list = []
        # ID 0:BackGround
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)
        color_image_list.append(bg_image)
        # ID 1:Road
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (255, 0, 0)
        color_image_list.append(bg_image)
        # ID 2:Curb
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)
        color_image_list.append(bg_image)
        # ID 4:Mark
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 255)
        color_image_list.append(bg_image)

        # Overlay segmentation map
        masks = segmentation_map.transpose(2, 0, 1)

        ls_image = []

        for index, mask in enumerate(masks):
            # Threshold check by score
            mask = np.where(mask > score, 0, 1)

            # Overlay
            # mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
            mask_image = np.full(mask.shape[:2], 255, dtype=mask.dtype)
            if index == 2:
                break


        return mask_image


    def _BirdsEyeViewTransfer(self, cv_bgr:np.ndarray, on_testview:bool=False, ratio_div_h:float=6/10, ratio_src_w:float=0.875, ratio_dst_w:float=0.2, ratio_view_h:float=0.3)->np.ndarray:
        
        '''
        Birds Eye View に画像を変換する。
        変更前、変更後の台形の形状から、変換行列を計算し、
        画像の中程から下までを切り取って形状を変換する。説明では画像の高さ,幅を(H,W)とする。
        args:
            cv_bgr: OpenCV BGR画像データ
            on_testview: Trueにすると、変換前の画像を上にくっつけて出力する
            ratio_div_h: H * ratio_div_h から下を切り取って画像を変換する
            ratio_src_w: W * ratio_src_w を変換前の台形の下底とする。上底はW
            ratio_dst_w: W * ratio_dst_w を変換後の台形の下底とする。上底はW
            ratio_trans_h: H * ratio_trans_h を変換前、変換後の台形の高さとする。  
        return:
            cv_bgr_ipm: 変換後のOpenCV BGR画像データ
        '''

        H,W = cv_bgr.shape[:2]

        xw = int(W * ((1.0 - ratio_src_w)/2.0))
        xw2= int(W * ((1.0 - ratio_dst_w)/2.0))
        yh = int(H * ratio_view_h)

        __h_range = [int(H*ratio_div_h), int(H*ratio_div_h+yh)]

        # 高さの一定レンジだけを切り取る
        img = cv_bgr[__h_range[0]:__h_range[1], 0:W]

        src = np.float32([[xw, yh], [W-xw, yh], [0, 0], [W, 0]])
        dst = np.float32([[xw2, yh], [W-xw2, yh], [0, 0], [W, 0]])
        M = cv.getPerspectiveTransform(src, dst)

        transfer_img = cv.warpPerspective(img, M, (img.shape[1],img.shape[0]))
    #    print("shape ",img.shape[:2])
    #    print("shape ",transfer_img.shape[:2])

        if on_testview == True :
            transfer_img = cv.vconcat([img, transfer_img])

        return transfer_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)

    args = parser.parse_args()

    # Initialize video capture
    print("open video file", args.movie)
    cap = cv.VideoCapture(args.movie)
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv.CAP_PROP_FPS))
    print(cap.get(cv.CAP_PROP_FRAME_COUNT))

    rc_segmentation = RCsegmentation()

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            print("Capture end")
            break

        image = copy.deepcopy(frame)

        segment_image, elapsed_time = rc_segmentation.loop(image)

        cv.imshow('road-segmentation-adas-0001 Demo', segment_image)

        key = cv.waitKey(1)
        if key == 'q':  # ESC
            print("ESC")
            break

    cap.release()
    cv.destroyAllWindows()
