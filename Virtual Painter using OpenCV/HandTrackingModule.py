import cv2
import mediapipe as mp
import time

# BY : RASHI SALUJA

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for i in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, i, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.landmarkList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, width, channels = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                self.landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return self.landmarkList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for i in range(1,5):
            if self.landmarkList[self.tipIds[i]][2] < self.landmarkList[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    obj = HandDetector()  # Creating an object
    while True:
        success, img = cap.read()
        img = obj.findHands(img)
        landmarkList = obj.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        # To display it in the output
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
