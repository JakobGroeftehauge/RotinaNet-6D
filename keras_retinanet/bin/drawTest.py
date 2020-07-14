import cv2
import numpy as np
import os
import sys
  
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"
    
from ..utils.visualization import draw_box, draw_caption

print(cv2.__version__)
image = cv2.imread('LINEMOD/000001/rgb/000001.png')
box = [71, 30, 240, 80]
b = np.array(box).astype(int)
draw_box(image, b, color=(0,0,255))
cv2.imwrite('test.png',image)