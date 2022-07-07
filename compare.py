import cv2

import numpy as np


compare_list = [
    "./videos/high_iter_defense_keep4.mp4"#,
    #"C:/Users/uif08808/Downloads/test_defense.mp4",
    #"C:/Users/uif08808/Downloads/test_defense2.mp4",
    #"C:/Users/uif08808/Downloads/test_defense3.mp4"
]

videoCapList = []

for path in compare_list:
    videoCapList.append(cv2.VideoCapture(path))


frames = []

for videoCapture in videoCapList:
    frames.append(videoCapture.read())

width  = int(videoCapList[0].get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(videoCapList[0].get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("compare.mp4", fourcc, 30, (width * 2, height * 2))


count = 0
while all(success_and_frame[0] for success_and_frame in frames):
    if count > 230 and count < 250:
    # left_stacked = np.concatenate((frames[0][1], frames[1][1]), axis=0)
    # right_stacked = np.concatenate((frames[2][1], frames[3][1]), axis=0)

    # total = np.concatenate((left_stacked, right_stacked), axis=1)
    # writer.write(total)
        cv2.imwrite(f"./test/2bit{count}.jpg", frames[0][1])
        #break
    for i in range(len(videoCapList)):
        frames[i] = videoCapList[i].read()

    count += 1
    if count % 100 == 0:
        print(count)


writer.release()