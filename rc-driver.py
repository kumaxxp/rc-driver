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


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, expand dimensions, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze
    segmentation_map = result[0]
    segmentation_map = np.squeeze(segmentation_map)

    return segmentation_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,896',
    )

    args = parser.parse_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    score = args.score

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv.CAP_PROP_FPS))
    print(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            print("Capture end")
            break
        # debug_image = copy.deepcopy(frame)

        # 動画からキャプチャした画像を縮小する
        h = input_size[0]
        w = input_size[1]
        size = (w,h)
        resize_image = cv.resize(frame, size)

        # セグメンテーション実行
        segmentation_map = run_inference(
            onnx_session,
            input_size,
            resize_image,
        )

        elapsed_time = time.time() - start_time

        # セグメンテーションを元画像にオーバーレイ
        debug_image = draw_debug(
            debug_image,
            score,
            segmentation_map,
        )

        # 物体検出実行

        # 元画像cv_bgrをsrc_ptsを台形に切り取って、鳥観図に変換する        
#        debug_image = BirdsEyeViewTransfer(debug_image, True)
        debug_image = BirdsEyeViewTransfer(debug_image, True, ratio_div_h=6/8, ratio_dst_w=0.20)

        # セグメンテーションに要した時間をオーバーレイ
        cv.putText(debug_image,
                "Elapsed : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                cv.LINE_AA)


        key = cv.waitKey(1)
        if key == 'q':  # ESC
            print("ESC")
            break
        cv.imshow('road-segmentation-adas-0001 Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, score, segmentation_map):
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


def BirdsEyeViewTransfer(cv_bgr:np.ndarray, on_testview:bool=False, ratio_div_h:float=6/10, ratio_src_w:float=0.875, ratio_dst_w:float=0.2, ratio_view_h:float=0.3)->np.ndarray:
    
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

    img = cv_bgr[int(H*ratio_div_h):H, 0:W]

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
    main()
