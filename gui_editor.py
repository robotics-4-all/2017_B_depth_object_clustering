import cv2
import numpy as np
import time
import yaml

import clusterer
import metaproccessor

def nothing(x):
    pass

# Read the parameters from the yaml file
with open("conf.yaml", 'r') as stream:
  try:
    doc = yaml.load(stream)
  except yaml.YAMLError as exc:
    print(exc)

rgbname = doc["images"]["rgbname"]
depthname = doc["images"]["depthname"]
clusteredname = doc["images"]["clusteredname"]

# Create a window with initial RGB and Depth images
rgbimg = cv2.imread(rgbname)
depthimg = cv2.imread(depthname)
img = np.concatenate((rgbimg, depthimg), axis=1)

cv2.namedWindow('image')

# Create trackbars for parameters of clusterer
cv2.createTrackbar('Clusters','image',2,21,nothing)
cv2.createTrackbar('Depth Weight1','image',0,9,nothing) # no float permitted
cv2.createTrackbar('Depth Weight2','image',0,16,nothing)# no float permitted 
#~ cv2.createTrackbar('Coord Weight','image',0,1,nothing)
cv2.createTrackbar('Depth ThresUp','image',0,255,nothing)
cv2.createTrackbar('Depth ThresDown','image',0,255,nothing)

# Set the default values fot the trackbars
cv2.setTrackbarPos('Clusters','image', doc["clustering"]["number_of_clusters"])
cv2.setTrackbarPos('Depth Weight1','image', 4)
cv2.setTrackbarPos('Depth Weight2','image', 2)
#~ cv2.setTrackbarPos('Coord Weight','image', doc["clustering"]["coordinates_weight"])
cv2.setTrackbarPos('Depth ThresUp','image', doc["clustering"]["depth_thresup"])
cv2.setTrackbarPos('Depth ThresDown','image', doc["clustering"]["depth_thresdown"])

print "Press ENTER to start or Esc to exit."
while(1):
  cv2.imshow('image',img)
  k = cv2.waitKey(1) & 0xFF
  if k == 27:
    break
    
  if k == 13:
    # Get current positions of trackbars
    nclusters = cv2.getTrackbarPos('Clusters','image')
    depth_weightn = cv2.getTrackbarPos('Depth Weight1','image')
    depth_weightp = cv2.getTrackbarPos('Depth Weight2','image')
    if depth_weightn == 0:
      depth_weight = 0
    else:
      depth_weight = depth_weightn ** (-depth_weightp)
    coord_weight = 0
    depth_thresup = cv2.getTrackbarPos('Depth ThresUp','image')
    depth_thresdown = cv2.getTrackbarPos('Depth ThresDown','image')
    
    # Apply the proccesing functions
    start_time = time.time()
    img = clusterer.clusterer(rgbname,depthname,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown)
    img = metaproccessor.metaproccessor(clusteredname,rgbname,depthname,nclusters,150)
    elapsed_time = time.time() - start_time
    print "Object detection is done in time:", elapsed_time,"s!"
    print "Press ENTER to start or Esc to exit."

cv2.destroyAllWindows()
