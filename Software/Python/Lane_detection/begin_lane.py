#!/usr/bin/python3
# Author: Ron Haber
# Date: 16.1.2021
# This script will help with lane detection and determine the correct params

import os, sys
import re, math
import cv2
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

def ReadImage(image_path):
    image = cv2.imread(image_path)
    return image

def EdgeDetection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def RegionOfInterest(img, vertices):
    mask = np.zeros_like(img)
    # gets the number of colour channels
    match_mask_colour = 255
    cv2.fillPoly(mask, vertices, match_mask_colour)
    # print(mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def GetImageVertices(image):
    height = image.shape[0]
    width = image.shape[1]
    vertices = [
        (0, int(0.77*height)), (int(width/2), int(height/2.1)), (width, int(0.77*height))
    ]
    output = np.array([vertices])
    return output

def DrawLines(img, lines, color=[255, 0, 0], thickness=5):
    if(lines is None):
        return
    img = np.copy(img)
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1,y1), (x2,y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 1.0)
    return img

def SeparateLines(image, lines):
    min_y = int(image.shape[0] * 0.476)
    max_y = (image.shape[0] * 77)
    max_x = image.shape[1]
    left_line_x = [0, 0]
    left_line_y = [0, 0]
    right_line_x = [max_x, 0]
    right_line_y = [max_x, 0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                slope = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                slope = 0 # disregard an invalid line
        if(math.fabs(slope) < 0.5):
            continue
        if(slope <=0): # left group since the slope is negative
            if(x1 > left_line_x[0]):
                left_line_x = [x1, x2]
                left_line_y = [y1, y2]
            # left_line_x.extend([x1, x2])
            # left_line_y.extend([y1, y2])
        else: # right group
            if(x1 < right_line_x[0]):
                right_line_x = [x1, x2]
                right_line_y = [y1, y2]
            # right_line_x.extend([x1, x2])
            # right_line_y.extend([y1, y2])
    
    # print(left_line_x)
    # print(left_line_y)
    # print(right_line_x)
    # print(right_line_y)
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    right_x_start = int(poly_right(max_y)) 
    right_x_end = int(poly_right(min_y))
    return [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]]    

def ApplyAll(image):
    vertices = GetImageVertices(image)
    canny = EdgeDetection(image)
    cropped_image = RegionOfInterest(canny, vertices)
    lines = cv2.HoughLinesP(
        cropped_image, 
        rho=6, 
        theta=np.pi/60, 
        threshold=160, 
        lines=np.array([]), 
        minLineLength=250, 
        maxLineGap=190
    )
    adjusted_lines = SeparateLines(image, lines)
    print(adjusted_lines)
    combo_image = DrawLines(image, adjusted_lines)
    return combo_image

image = ReadImage('autobahn_fahren.png')
combo_image = ApplyAll(image)
plt.figure()
plt.imshow(combo_image)
plt.show()
