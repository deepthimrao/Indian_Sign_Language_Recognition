'''perform palm localization'''

import cv2
import mediapipe as mp
import time
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

def get_landmark_points(image):
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    return results.multi_handedness
    
    
def main():
    path = 'dataset/'
    data = os.listdir('dataset/')
    print(data)
    for label in data:
        images_path = path+label+'/'
        for img in os.listdir(images_path):
            image_name = images_path+img
            image = cv2.imread(image_name)
            cv2.imshow('test',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(label,len(get_landmark_points(image)))
            break
            
    
main()