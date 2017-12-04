#!/usr/bin/python
import cv2
import numpy as np
import yaml

with open("conf.yaml", 'r') as stream:
  try:
    doc = yaml.load(stream)
  except yaml.YAMLError as exc:
    print(exc)
    
nclusters = doc["clustering"]["number_of_clusters"]
rgbname = doc["images"]["rgbname"]
depthname = doc["images"]["depthname"]

img = cv2.imread('Results/result_cur.png')
rgbimg = cv2.imread(rgbname)
imgdepth = cv2.imread(depthname,cv2.IMREAD_GRAYSCALE)

coldict = {
'[0]': [230, 25, 75],
'[1]': [60, 180, 75],
'[2]': [255, 225, 25],
'[3]': [0, 130, 200],
'[4]': [245, 130, 48],
'[5]': [145, 30, 180],
'[6]': [70, 240, 240],
'[7]': [240, 50, 230],
'[8]': [210, 245, 60],
'[9]': [250, 190, 190],
'[10]': [0, 128, 128],
'[11]': [230, 190, 255],
'[12]': [170, 110, 40],
'[13]': [255, 250, 200],
'[14]': [128, 0, 0],
'[15]': [170, 255, 195],
'[16]': [128, 128, 0],
'[17]': [255, 215, 180],
'[18]': [0, 0, 128],
'[19]': [128, 128, 128],
'[20]': [255, 255, 255],
'[21]': [0, 0, 0]
}

imgproc = img
height, width, channels = img.shape
overall_mask = np.zeros((height,width), np.uint8) # blank mask

for i in range(0, nclusters):
  string_index = '['+str(i)+']'
  desiredcolor = coldict[string_index]  

  desiredcolorarray = np.array(desiredcolor, dtype = "uint8")
  maskinit = cv2.inRange(img, desiredcolorarray, desiredcolorarray)
  kernelclosing = np.ones((10,10),np.uint8)
  kernelopening = np.ones((5,5),np.uint8)
  kernelgrad = np.ones((5,5),np.uint8)
  kernelero = np.ones((3,3),np.uint8)
  
  mask = cv2.morphologyEx(maskinit, cv2.MORPH_OPEN, kernelopening)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelclosing)
  mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernelgrad)
  mask = cv2.erode(mask,kernelero,iterations = 1)
  overall_mask = np.bitwise_or(mask, overall_mask)

imgproc = cv2.bitwise_and(rgbimg, rgbimg, mask = cv2.bitwise_not(overall_mask))

vis1 = np.concatenate((rgbimg, cv2.cvtColor(imgdepth,cv2.COLOR_GRAY2RGB)), axis=1)
vis2 = np.concatenate((img, imgproc), axis=1)
finalvis = np.concatenate((vis1, vis2), axis=0)
cv2.imshow("Image after processing", finalvis)
cv2.waitKey(0)
#~ raw_input("Press Esc and Enter to end...")
cv2.destroyAllWindows()
