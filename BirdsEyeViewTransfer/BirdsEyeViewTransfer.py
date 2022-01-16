#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

# 

import cv2 as cv
import numpy as np

class BirdsEyeViewTransfer:

    def __init__(self) -> None:
        pass

    # これを関数として実装する意味はあまりなさそう
    def resize(self, image):

        # 動画からキャプチャした画像を縮小する
        h = self.input_size[0]
        w = self.input_size[1]
        size = (w,h)
        resize_image = cv.resize(image, size)

        return resize_image

    def Transfer(self, cv_bgr:np.ndarray, on_testview:bool=False, ratio_div_h:float=6/10, ratio_src_w:float=0.875, ratio_dst_w:float=0.2, ratio_view_h:float=0.3)->np.ndarray:
        
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

        h_range = [int(H*ratio_div_h), int(H*ratio_div_h+yh)]

        # 高さの一定レンジだけを切り取る
        img = cv_bgr[h_range[0]:h_range[1], 0:W]

        src = np.float32([[xw, yh], [W-xw, yh], [0, 0], [W, 0]])
        dst = np.float32([[xw2, yh], [W-xw2, yh], [0, 0], [W, 0]])
        M = cv.getPerspectiveTransform(src, dst)

        transfer_img = cv.warpPerspective(img, M, (img.shape[1],img.shape[0]))

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

    # 動画化して保存する
    size = (1280,768)
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    video  = cv.VideoWriter('ImgVideo.mp4', fourcc, cap.get(cv.CAP_PROP_FPS), size)

    birds_eye_view_transfer = BirdsEyeViewTransfer()

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            print("Capture end")
            break

        image = copy.deepcopy(frame)

        # 妥当な大きさにリサイズ
        resize_image = cv.resize(image, size)

        # 鳥観図に変換
        birds_eye_image = birds_eye_view_transfer.Transfer(resize_image,True)
        video.write(cv.resize(birds_eye_image, size))

        cv.imshow('road-segmentation-adas-0001 Demo', birds_eye_image)

        key = cv.waitKey(1)
        if key == 'q':  # ESC
            print("ESC")
            break


    video.release()
    cap.release()
    cv.destroyAllWindows()
