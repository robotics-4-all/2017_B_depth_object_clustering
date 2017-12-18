#!/usr/bin/python
import cv2
import numpy as np
import math
import yaml

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: 
    return ()
  return (x, y, w, h)
  
def is_sameplane(points1, points2,threshold):
  # These two vectors are in the plane
  v1 = points1[2] - points1[0]
  v2 = points1[1] - points1[0]
  # Cross product is a vector normal to the plane
  cp = np.cross(v1, v2)
  # This evaluates a * x3 + b * y3 + c * z3 - d = 0
  abcd = np.append(cp, -np.dot(cp, points1[2]))
  # Put the points2 in the equation. The first one is common, thus ignore it.
  return (sum(np.multiply(abcd, np.append(points2[1],1))) <= threshold) or (sum(np.multiply(abcd, np.append(points2[2],1))) <= threshold)
  
def removearray(L,arr):
  ind = 0
  size = len(L)
  while ind != size and not np.array_equal(L[ind],arr):
      ind += 1
  if ind != size:
      L.pop(ind)
  else:
      raise ValueError('Array not found in list.')
  

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
  
  prefinal_contours = list()
  for i in range(0, nclusters):
    desiredcolor = coldict[i]  

    desiredcolorarray = np.array(desiredcolor, dtype = "uint8")
    maskinit = cv2.inRange(img, desiredcolorarray, desiredcolorarray)
    kernelclosing = np.ones((10,10),np.uint8)
    kernelopening = np.ones((5,5),np.uint8)
    kernelgrad = np.ones((5,5),np.uint8)
    kernelero = np.ones((3,3),np.uint8)
    
    # Apply morphological operators to get the contour of all labeled areas
    mask = cv2.morphologyEx(maskinit, cv2.MORPH_OPEN, kernelopening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelclosing)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernelgrad)
    mask = cv2.erode(mask,kernelero,iterations = 1)
    mask = cv2.erode(mask,np.ones((2,2),np.uint8),iterations = 1)
    
    image, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours[0:len(contours):2]: # TODO fix this, every contour is double
      if cv2.contourArea(c) > minsize:
        prefinal_contours.append(c)
    overall_mask = np.bitwise_or(mask, overall_mask)

  final_contours = list(prefinal_contours)
  # Check for intersection of bounding boxes
  for i in range(len(prefinal_contours)-1):
    box1 = cv2.boundingRect(prefinal_contours[i])
    for j in range(i+1, len(prefinal_contours)):
      box2 = cv2.boundingRect(prefinal_contours[j])
      intersec = intersection(box1,box2)
      # Check if there is an intersection and it's larger than the half of the minimum of bounding boxes
      if intersec != () and intersec[2]*intersec[3] > 0.5 * min((box1[2]*box1[3]),(box2[2]*box2[3])):
        center1 = np.asarray(tuple(map(lambda x, y: x + y/2, box1[0:2], box1[2:24])))
        center2 = np.asarray(tuple(map(lambda x, y: x + y/2, box2[0:2], box2[2:24])))
        dist1 = [center1[0] - intersec[0], center1[1] - intersec[1]]
        dist2 = [center2[0] - intersec[0], center2[1] - intersec[1]]
        # Vectors that lead from the intersection point to the center of bounding box
        unitvect1 = np.sign(dist1)
        unitvect2 = np.sign(dist2)
        pixel1 = np.asarray(intersec[0:2]) + unitvect1 * 15
        pixel2 = np.asarray(intersec[0:2]) + unitvect2 * 15
        
        # Compare the depths of bounding box in a neighborhood and compare it with a threshold
        if abs(int(imgdepth[pixel1[1]][pixel1[0]]) - int(imgdepth[pixel2[1]][pixel2[0]])) < 1:
          # Find the smallest bounding box, check if it is already removed and remove it.
          if cv2.contourArea(prefinal_contours[i]) < cv2.contourArea(prefinal_contours[j]) and any((np.array_equal(prefinal_contours[i],x)) for x in final_contours):
            removearray(final_contours, prefinal_contours[i])
          elif cv2.contourArea(prefinal_contours[i]) >= cv2.contourArea(prefinal_contours[j]) and any((np.array_equal(prefinal_contours[j],x)) for x in final_contours):
            removearray(final_contours, prefinal_contours[j])
  for c in final_contours:
    object_counter += 1
    # Get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # Draw rectangle to visualize the bounding rect with color
    cv2.rectangle(imgproc, (x, y), (x+w, y+h), coldict[object_counter], 1)
    cv2.putText(imgproc, str(object_counter), (x,y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, coldict[object_counter],1)
        
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
