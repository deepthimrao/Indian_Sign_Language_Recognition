import cv2
import mediapipe as mp
import time
import os

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def main():
    pTime = 0
    cTime = 0
    # cap = cv2.VideoCapture(0)
    detector = handDetector(mode=True)

    # while True:
    for file in os.listdir('dataset/B'):
        # success, img = cap.read()
        print(file)
        img = cv2.imread('dataset/B/'+file)
        # print(img)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        points = []
        for p in lmlist:
            points.append(p[1:])
        # print(points)
        if(len(points)>0):
            a = bounding_box(points)
            cv2.rectangle(img,a[0],a[1],(0,0,0),3)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()