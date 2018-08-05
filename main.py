# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:03:20 2018

@author: siyuan
"""

import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import time 
import os 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def make_kernal(n):
#    a, b = (n-1)/2, (n-1)/2
#    r = (n-1)/2
#    y,x = np.ogrid[-a:n-a, -b:n-b]
#    mask = x*x + y*y <= r*r
#    kernal = np.ones((n, n)).astype(np.uint8)
#    kernal[mask] = 255
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernal 

def calibration(img,background):
    M = np.load('warp_matrix.npy')
    rows,cols,cha = img.shape
    imgw = cv2.warpPerspective(img, M, (cols, rows))
    imgwc = imgw[12:,71:571,:]
    bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
    bg_imgwc = bg_imgw[12:,71:571,:]
    img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32),(25,25),30)
    img_bs = imgwc.astype(np.int32) - img_blur.astype(np.int32) + np.mean(img_blur) +20
#    print(np.mean(img_blur))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    blur = 255-rgb2gray(img_blur)
    blur = blur/np.max(blur)*(0.2+(np.mean(img_blur)/np.mean(img_blur)-1)*0.8)
    blur_op = 1-blur
    cl1 = clahe.apply(img_bs[:,:,0].astype(np.uint8))
    cl2 = clahe.apply(img_bs[:,:,1].astype(np.uint8))
    cl3 = clahe.apply(img_bs[:,:,2].astype(np.uint8))
    red = img_bs[:,:,0]*blur_op + cl1*blur
    green = img_bs[:,:,1]*blur_op + cl2*blur
    blue = img_bs[:,:,2]*blur_op + cl3*blur
    im_calibrated = np.dstack((red,green,blue))
    return im_calibrated,img_blur,imgwc

def calibration_v2(img,background):
    M = np.load('warp_matrix.npy')
    rows,cols,cha = img.shape
    imgw = cv2.warpPerspective(img, M, (cols, rows))
    imgwc = imgw[12:,71:571,:]
    bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
    bg_imgwc = bg_imgw[12:,71:571,:]
    img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32),(25,25),30)
    im_sub = imgwc/img_blur*100
    return im_sub,img_blur,imgwc



def creat_mask(im_cal,threshold):
    img_gray = rgb2gray(im_cal).astype(np.uint8)
    ret,thresh1 = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
#    plt.figure()
#    plt.imshow(img_gray)
#    plt.show()
#    plt.figure()
#    plt.imshow(thresh1)
#    plt.show()
    kernal1 = make_kernal(8)
    kernal2 = make_kernal(3)
    final_image2 = cv2.erode(thresh1, kernal1, iterations=1)
    final_image = cv2.dilate(final_image2, kernal2, iterations=1)
    return final_image

def find_dots(binary_image):
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 2
    params.maxThreshold = 255
    # Filter by Area.
#    params.filterByArea = False
#    params.minArea = 1
#    # Filter by Circularity
#    params.filterByCircularity = False
#    params.minCircularity = 0.1
#    # Filter by Convexity
#    params.filterByConvexity = False
#    params.minConvexity = 0.87
#    # Filter by Inertia
#    params.filterByInertia = False
#    params.minInertiaRatio = 0.01
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create()
    # Detect blobs.
    keypoints = detector.detect(binary_image.astype(np.uint8))
    return keypoints

            
def flow_calculate(keypoints2,x1_last,y1_last,trash_list):
    xy2, u, v, x2, y2 = [], [], [], [], [] 
    for i in range(len(keypoints2)): 
        xy2.append([keypoints2[i].pt[0],keypoints2[i].pt[1]])
        
    xy2 = np.array(xy2) 

    for i in range(len(x1_last)):
        distance = list(np.sqrt((x1_last[i] - np.array(xy2[:,0]))**2 + (y1_last[i] - np.array(xy2[:,1]))**2))
        min_index = distance.index(min(distance))
        u_temp = x1_last[i] - xy2[min_index,0]
        v_temp = y1_last[i] - xy2[min_index,1]

        if np.sqrt(u_temp**2+v_temp**2) > 10:
            u_temp = 0
            v_temp = 0
            x2.append(x1_last[i])
            y2.append(y1_last[i])
            trash_list.append(i)
        else:
            np.delete(xy2,min_index,0)
            x2.append(xy2[min_index,0])
            y2.append(xy2[min_index,1])

            
        u.append(u_temp)
        v.append(v_temp)
        
    for i in range(len(trash_list)):
        u[trash_list[i]] = 0
        v[trash_list[i]] = 0

    return x2,y2,u,v,trash_list

def dispOpticalFlow(im_cal,x,y,u,v):
        mask = np.zeros_like(im_cal)

        for i in range(len(x)):
             mask = cv2.line(mask, (int(x[i]-u[i]*2),int(y[i]-v[i]*2)),(int(x[i]),int(y[i])), [0, 80, 0], 2)

        img = cv2.add(im_cal/2,mask)

        cv2.imshow("force_flow",img.astype(np.uint8))
        cv2.waitKey(0)
     
#%%

if __name__ == "__main__":
    
    im_ref = cv2.imread('GS2_3416.png')
    raw_imag = cv2.imread('GS2_4560.png')
    im_cal_ref,img_blur,imgwc = calibration_v2(im_ref,im_ref)
    im_cal,img_blur,imgwc = calibration_v2(raw_imag,im_ref)
    
    final_image_ref = creat_mask(im_cal_ref,60)
    final_image = creat_mask(im_cal,60)
    keypoints_ref = find_dots(final_image_ref)
    keypoints = find_dots(final_image)
    trash_list = []
    
    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(keypoints_ref)):
        x1.append(keypoints_ref[i].pt[0])
        y1.append(keypoints_ref[i].pt[1])
    
    for i in range(len(keypoints)):
        x2.append(keypoints[i].pt[0])
        y2.append(keypoints[i].pt[1])
        
    x2,y2,u,v,trash_list = flow_calculate(keypoints,x1,y1,trash_list)
    trash_list = sorted(set(trash_list))
    u = np.array(u)
    v = np.array(v)
    u[trash_list] = 0 
    v[trash_list] = 0             
        
    dispOpticalFlow(im_cal,x2,y2,u,v)
    
    
    
