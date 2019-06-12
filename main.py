#! /usr/bin/env python
# -*- coding: utf-8 -*-

import colorsys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from timeit import default_timer as timer
from getch import getch, pause

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=1

#- Added
import cv2
cap = cv2.VideoCapture(0)
camera_scale = 1.

import yoloobject

def detect_img(yolo):
    while True:
        ret, image = cap.read()
        if cv2.waitKey(10) == 27:
            break
        h, w = image.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)
        image = cv2.resize(image, (rw, rh))
        image = image[:,:,(2,1,0)]
        image = Image.fromarray(image)
        r_image = yolo.detect_image(image)
        out_img = np.array(r_image)[:,:,(2,1,0)]
        cv2.imshow("cam", np.array(out_img))
        cv2.waitKey(100)
    yolo.close_session()

if __name__ == '__main__':
    detect_img(yoloobject.YOLO())