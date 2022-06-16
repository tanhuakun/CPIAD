import cv2
import numpy as np


path="./videos/germany-1.mp4"
videoCap = cv2.VideoCapture(path)


width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = videoCap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("test.mp4", fourcc, fps, (width, height))


def bitwise_and_compose(mask):
    def func(value):
        return value & mask
    return func
    

success, frame = videoCap.read()

map_func = bitwise_and_compose(int("11110000", 2))

while success:
    writer.write(map_func(frame))
    success, frame = videoCap.read()

writer.release()
