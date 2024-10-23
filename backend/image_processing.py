import cv2
import numpy as np

def upscale_image(input_path, output_path):
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Upscale image using bicubic interpolation
    height, width = img.shape[:2]
    upscaled = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)

    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
