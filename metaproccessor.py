#!/usr/bin/python
import cv2
import numpy as np



img = cv2.imread('./result_cur.png')
#~ img = cv2.medianBlur(img,5)
size_window = 3
imgproc =  cv2.medianBlur(img,size_window)
imgproc =  cv2.medianBlur(imgproc,size_window)
imgproc =  cv2.medianBlur(imgproc,size_window)
imgproc =  cv2.medianBlur(imgproc,size_window)


vis = np.concatenate((img, imgproc), axis=1)
cv2.imshow('result after median filter',vis)
cv2.waitKey(0)
raw_input("Press Esc and Enter to end...")
cv2.destroyAllWindows()
