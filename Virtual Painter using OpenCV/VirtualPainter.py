import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import imutils

# BY : RASHI SALUJA

path = "Images"
imageList = os.listdir(path)
# print(imageList)
overLayList = []

for i in imageList:
    image = cv2.imread(f'{path}/{i}')
    overLayList.append(image)

header = overLayList[2]  # setting red paint color to be selected by default
drawColor = (0, 0, 255)
xp, yp = 0, 0

brushThickness = 15
eraserThickness = 50

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

obj = htm.HandDetector(detectionConfidence=0.85)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Finding the hand landmarks
    img = obj.findHands(img)
    lmList = obj.findPosition(img, draw=False)

    if len(lmList) != 0:

        # getting the landmark of index finger
        x1, y1 = lmList[8][1:]
        # getting the landmark of middle finger
        x2, y2 = lmList[12][1:]

        # Checking which fingers are up
        fingers = obj.fingersUp()
        # print(fingers)

        # Selection mode (when two fingers are up)
        if fingers[1] and fingers[2]:
            # print('Selection')
            xp, yp = 0, 0

            # Checking for the click
            if y1 < 270:
                if 250 < x1 < 450:
                    header = overLayList[2]  # Red
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overLayList[1]  # Green
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overLayList[3]  # Yellow
                    drawColor = (0, 255, 255)
                elif 1050 < x1 < 1200:
                    header = overLayList[0]  # Eraser
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

        # Drawing mode (when one finger is up)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print('Drawing')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:270, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
