# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:32:08 2019

@author: Helen
"""
import cv2
import os

predict_list = "E:/A_paper_thesis/00_modify_DeepLabV3plus/DeepLabv3plus/inference_data/pred_results"
gt_list = "E:/A_paper_thesis/00_modify_DeepLabV3plus/DeepLabv3plus/inference_data/GT"
lungimg_list = "E:/A_paper_thesis/00_modify_DeepLabV3plus/DeepLabv3plus/inference_data/Lung_img"
filename_id = os.listdir("E:/A_paper_thesis/00_modify_DeepLabV3plus/DeepLabv3plus/inference_data/Lung_img")

output = "E:/A_paper_thesis/00_modify_DeepLabV3plus/DeepLabv3plus/inference_data/boundary_results/"

#doing for mnay image by using image id
# draw contour for ground truth
for i, img_id in enumerate(filename_id):
  gt = cv2.imread(os.path.join(gt_list, img_id))  
  imgray_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
  imgray_gt = cv2.medianBlur(imgray_gt, ksize=7)
  
  ret, thresh_gt = cv2.threshold(imgray_gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  _, contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
#  new = cv2.drawContours(im,contours,len(contours)-1,(0,255,0),2)

# draw contour for predict output
  pred = cv2.imread(os.path.join(predict_list, img_id))
  pred1 = cv2.cvtColor(pred, cv2.COLOR_RGBA2BGR)
  crop_img = pred1[10:379, 10:379]
  pred2 = cv2.resize(crop_img, (512,512))
  imgray_pred = cv2.cvtColor(pred2, cv2.COLOR_BGR2GRAY)
  ret, thresh_pred = cv2.threshold(imgray_pred, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  _, contours_pred, _ = cv2.findContours(thresh_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

## wapping contour of ground truth and predict to original image
  im_ori = cv2.imread(os.path.join(lungimg_list, img_id))
  img_outname =img_id[:-4]
  
  cv2.drawContours(im_ori,contours_gt,len(contours_gt)-1,(0,255,0),1)
  cv2.drawContours(im_ori,contours_pred,len(contours_pred)-1,(0,0,255),1)
  cv2.imwrite(os.path.join(output) + img_outname + "_result.jpg", im_ori)
  #cv2.imshow('predict', predict)
  ##
#cv2.imshow('results', im_ori[0])
cv2.waitKey(0)

