import cv2

import numpy as np
# open camera
capture = cv2.VideoCapture(2)

# Initializing current time and precious time for calculating the FPS
while True:

    ret, frame = capture.read()

    print(frame.shape)

    if ret:
        cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()