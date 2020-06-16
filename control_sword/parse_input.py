import os
from typing import List

import cv2

cwd = os.getcwd()
train_data_path = cwd + "/data/train/"
parsed_train_data_path = cwd + "/data/parsed/"


def parse_video_into_frames(input_path, output_path):
    video = cv2.VideoCapture(input_path)
    success, image = video.read()

    count = 0
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    while success:
        path = os.path.join(output_path, "frame{}.jpg".format(count))
        if not cv2.imwrite(path, image):  # save frame as JPEG file
            raise Exception("Could not write image")
        success, image = video.read()
        count += 1


def iterate_directory(directory: str, extension: str) -> List[str]:
    return list(filter(lambda x: x.endswith(extension), os.listdir(directory)))


if __name__ == "__main__":
    # training process
    for idx, video_name in enumerate(iterate_directory(train_data_path, "mp4")):
        output_folder = os.path.join(train_data_path, str(idx + 1))
        print("Parsing video {} into folder {}".format(video_name, output_folder))
        parse_video_into_frames(os.path.join(train_data_path, video_name), output_folder)
