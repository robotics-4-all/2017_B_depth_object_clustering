#!/usr/bin/python
import cv2
import numpy as np
import yaml

def metaproccessor(img,rgbimg,imgdepth,nclusters,minsize):

  with open("../cfg/conf.yaml", 'r') as stream:
    try:
      doc = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  coldict = doc["clustering"]["coldict"]

  imgproc = rgbimg.copy()
  height, width, channels = img.shape
  overall_mask = np.zeros((height,width), np.uint8) # blank mask
  object_counter = 0

  for i in range(0, nclusters):
    desiredcolor = coldict[i]  

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
    mask = cv2.erode(mask,np.ones((2,2),np.uint8),iterations = 1)
    
    image, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours[0:len(contours):2]: # TODO fix this, every contour is double
      if cv2.contourArea(c) > minsize:
        object_counter += 1
        # Get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # Draw rectangle to visualize the bounding rect with label-color
        cv2.rectangle(imgproc, (x, y), (x+w, y+h), coldict[i], 1)
        cv2.putText(imgproc, str(object_counter), (x,y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, coldict[i],1)
      
    overall_mask = np.bitwise_or(mask, overall_mask)
  print "Number of objects detected:", object_counter
  img_mask = cv2.bitwise_and(rgbimg, rgbimg, mask = cv2.bitwise_not(overall_mask))
  vis1 = np.concatenate((rgbimg, cv2.cvtColor(imgdepth,cv2.COLOR_GRAY2RGB), img), axis=1)
  vis2 = np.concatenate((cv2.cvtColor(overall_mask,cv2.COLOR_GRAY2RGB),img_mask,imgproc), axis=1)
  finalvis = np.concatenate((vis1, vis2), axis=0)
  return finalvis
  
if __name__ == '__main__':
  with open("../cfg/conf.yaml", 'r') as stream:
    try:
      doc = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  nclusters = doc["clustering"]["number_of_clusters"]
  rgbname = doc["images"]["rgbname"]
  depthname = doc["images"]["depthname"]
  clusteredname = doc["images"]["clusteredname"]
  img = cv2.imread(clusteredname)
  rgbimg = cv2.imread(rgbname)
  imgdepth = cv2.imread(depthname,cv2.IMREAD_GRAYSCALE)
  
  vis = metaproccessor(img,rgbimg,imgdepth,nclusters,150)
  cv2.imshow("Image after processing", vis)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
