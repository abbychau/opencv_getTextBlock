import cv2
import pytesseract as tess
from PIL import Image
import numpy as np


# Load image, grayscale, Gaussian blur, adaptive threshold
image = cv2.imread('t.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9,9), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

# Dilate to combine adjacent text contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
dilate = cv2.dilate(thresh, kernel, iterations=4)

# Find contours, highlight text areas, and extract ROIs
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]




ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10000:
        x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        ROI = image[y:y+h, x:x+w]

        gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        ret,ROI = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

        # cv2.bitwise_not(binary,binary)
        # textImage = Image.fromarray(binary)

        text=tess.image_to_string(ROI, lang='chi_sim')
        print("ROI: %s"%text)

        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
# cv2.imshow('image', image)
# cv2.waitKey()