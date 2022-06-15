import cv2

import numpy as np


vid1 = "./videos/germany_detect_attack2.mp4"
vid2 = "./videos/germany_detect_attack3.mp4"



videoCap1 = cv2.VideoCapture(vid1)
videoCap2 = cv2.VideoCapture(vid2)

success1, frame1 = videoCap1.read()
success2, frame2 = videoCap2.read()


count = 0

while success1 and success2:
    stacked = np.concatenate((frame1, frame2), axis=0)
    cv2.imwrite(f"./test/{count}.jpg", stacked)
    success1, frame1 = videoCap1.read()
    success2, frame2 = videoCap2.read()
    count += 1
    if count % 100 == 0:
        (count)