# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:32:08 2019

@author: Helen
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np

maskpath = "E:/A_paper_thesis/Paper00_ThesisCode/DeepLabv3plus_lungCancer/dataset/LungCancer_mainData/Testing/GT_testA/combine_pre/IMG-0050-00115.jpg"
imgpath = "E:/A_paper_thesis/Paper00_ThesisCode/DeepLabv3plus_lungCancer/dataset/LungCancer_mainData/Testing/lung_testA/IMG-0050-00115.jpg"

predict = "E:/A_paper_thesis/Paper00_ThesisCode/DeepLabv3plus_lungCancer/dataset/inference_output/IMG-0050-00115_mask.png"


# draw contour for ground truth
im = cv2.imread(maskpath)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imgray = cv2.medianBlur(imgray, ksize=7)

ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

new = cv2.drawContours(im,contours,len(contours)-1,(0,255,0),2)

# draw contour for predict output
predict = cv2.imread(predict)
rgb = cv2.cvtColor(predict, cv2.COLOR_RGBA2BGR)
crop_img = rgb[10:379, 10:379]
rgb1 = cv2.resize(crop_img, (512,512))
imgray1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(imgray1, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

_, contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

## wapping contour of ground truth and predict to original image
im_ori = cv2.imread(imgpath)

cv2.drawContours(im_ori,contours,len(contours)-1,(0,255,0),1)
cv2.drawContours(im_ori,contours1,len(contours1)-1,(0,0,255),1)
cv2.imwrite("hello.png",im_ori)
cv2.imshow('results', im_ori)
cv2.imshow('predict', predict)
#
cv2.waitKey()

