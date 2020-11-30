import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    # slope, intercept = line_parameters
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            try:
                slope, intercept = parameters
            except TypeError:
                slope, intercept = 0.001, 0
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_temp = cv2.Canny(blur, 50, 150)
    # print('pass through canny')
    return canny_temp


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=7)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0.1*width, height), (0.9*width, height), (width/2, height/2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    # print('Pass through roi')
    return masked_image


"""
# 537...120, 900....... 480, 300
img = cv2.imread('Lane.jpeg')
lane_image = np.copy(img)
canny_base = canny(lane_image)
cropped_image = region_of_interest(canny_base)
cv2.imshow('crop', cropped_image)
cv2.waitKey(0)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# print(lines)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=30)
print(lines)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('Result', cropped_image)
cv2.imshow('Final Tracked Line', combo_image)
# cv2.imshow('Line Image', line_image)
cv2.waitKey(0)
"""
cap = cv2.VideoCapture("road-video-master/test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    # lane_image = np.copy(frame)
    if _:
        canny_base = canny(frame)
        cropped_image = region_of_interest(canny_base)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=70, maxLineGap=250)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        # cv2.imshow('Result', cropped_image)
        cv2.imshow('Final Tracked Line', combo_image)
        # cv2.imshow('Line Image', line_image)
        cv2.waitKey(1)

cap.release()