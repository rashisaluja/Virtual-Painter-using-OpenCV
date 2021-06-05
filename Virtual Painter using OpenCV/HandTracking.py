import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # To check if the result is working fine
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for i in result.multi_hand_landmarks:
            for id, lm in enumerate(i.landmark):
                # print(id,lm)
                height, width, channels = img.shape

                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(id, cx, cy)

                # to detect any id
                # if id == 0:
                cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)



            mpDraw.draw_landmarks(img, i, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # To display it in the output
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
