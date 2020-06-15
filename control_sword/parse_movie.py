import os
import cv2

video = cv2.VideoCapture('D:/ObliczeniaInteligentne2/control_sword/data/train/in1.mp4')
success, image = video.read()
count = 0

print(video, success)

while success:
    path = "D:/ObliczeniaInteligentne2/control_sword/parsed/frame{}.jpg".format(count)
    if not cv2.imwrite(path, image):  # save frame as JPEG file
        raise Exception("Could not write image")
    success, image = video.read()
    count += 1
