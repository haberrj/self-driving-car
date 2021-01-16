#!/usr/bin/python3
# Author: Ron Haber
# Date: 16.1.2021
# This script will detect lanes from pics and video streams
# will detect web stream data eventually

import os, sys
import re, math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point


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
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def SeparateLines(image, lines):
    min_y = int(image.shape[0] * 0.476)
    max_y = (image.shape[0] * 77)
    rx1 = []
    rx2 = []
    ry1 = []
    ry2 = []
    lx1 = []
    lx2 = []
    ly1 = []
    ly2 = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                slope = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                slope = 0 # disregard an invalid line
        if(math.fabs(slope) < 0.5):
            continue
        if(slope <=0): # left group since the slope is negative
            lx1.append(x1)
            lx2.append(x2)
            ly1.append(y1)
            ly2.append(y2)
        else: # right group
            rx1.append(x1)
            rx2.append(x2)
            ry1.append(y1)
            ry2.append(y2)
    try:
        right_line_x = [(sum(rx1)/len(rx1)), (sum(rx2)/len(rx2))] # will get an average
        right_line_y = [(sum(ry1)/len(ry1)), (sum(ry2)/len(ry2))]
        left_line_x = [(sum(lx1)/len(lx1)), (sum(lx2)/len(lx2))] # will get an average
        left_line_y = [(sum(ly1)/len(ly1)), (sum(ly2)/len(ly2))]
    except ZeroDivisionError:
        right_line_x = [0, 0] # will get an average
        right_line_y = [0, 0]
        left_line_x = [0, 0] # will get an average
        left_line_y = [0, 0]

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
    return [left_x_start, max_y, left_x_end, min_y], [right_x_start, max_y, right_x_end, min_y]    

def FindIntersectionPoint(lines):
    lineA = lines[0][0]
    lineB = lines[0][1]
    xA = (lineA[0],lineA[1])
    yA = (lineA[2],lineA[3])
    xB = (lineB[0], lineB[1])
    yB = (lineB[2], lineB[3])
    # uses shapely
    line1 = LineString([xA, yA])
    line2 = LineString([xB, yB])
    int_pt = line1.intersection(line2)
    try:
        poi = int_pt.x, int_pt.y
    except AttributeError:
        poi = 0, 0
    return poi

def CompareIntersectionPoint(image, lines):
    height = image.shape[0]
    width = image.shape[1]
    poi = FindIntersectionPoint(lines)
    if(poi[1] > (height*0.57)):
        return False
    else:
        return True

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
    return adjusted_lines

def GetAverageLaneLine(lane_lines, count):
    right = [0, 0, 0, 0]
    left = [0, 0, 0, 0]
    for line in lane_lines:
        for i in range(0,4):
            right[i] += line[0][i]
            left[i] += line[1][i]
    new_right = [int(val / count) for val in right]
    new_left = [int(val / count) for val in left]
    output = [[new_right, new_left]]
    return output

def PlayVideo(video):
    cnt = 0
    average_lane_lines = []
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        _, frame = cap.read()
        try:
            if(cnt < 10):
                average_lane_lines.append(ApplyAll(frame))
                cnt += 1
                continue
            else:
                average_lane_lines.append(ApplyAll(frame))
                average_lane_lines.pop(0) # remove the first value
        except ValueError:
            continue
        except TypeError:
            continue
        average = GetAverageLaneLine(average_lane_lines, cnt)
        if(CompareIntersectionPoint(frame, average) == False):
            combo_image = frame
        else:
            try:
                combo_image = DrawLines(frame, average)
                print(average)
            except TypeError:
                combo_image = frame
        cv2.imshow("results", combo_image)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_direc = '/home/ronhaber/Documents/Self_Driving/Lane_Detection/autobahn_fahren.mp4'
    PlayVideo(video_direc)



