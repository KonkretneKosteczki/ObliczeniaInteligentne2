import cv2
from PIL import Image

ALEXNET_INPUT_SIZE = 224
KEYPOINT_DETECTION_SIZE = 96

def divide(mp4filename, output_images_filename, crop_to_square_of_size=None):
  vidcap = cv2.VideoCapture(mp4filename)

  count = 0
  while True:
    success,image = vidcap.read()
    if not success:
      break
    if crop_to_square_of_size:
      img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      im_pil = Image.fromarray(img)
      im_pil.resize((crop_to_square_of_size,crop_to_square_of_size)).save(f'{output_images_filename}{count}.jpg', 'JPEG')
    else:
      cv2.imwrite(f"{output_images_filename}{count}.jpg", image)
    count += 1

if __name__ == '__main__':
  divide('train/in1.mp4', 'ink/in1_', KEYPOINT_DETECTION_SIZE)

