import cv2
import numpy as np
import clusterer
import metaproccessor
import yaml

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

# Create a window with RGB and Depth images
rgbimg = cv2.imread(rgbname)
depthimg = cv2.imread(depthname)
img = np.concatenate((rgbimg, depthimg), axis=1)

cv2.namedWindow('image')

# Create trackbars for parameters of clusterer
cv2.createTrackbar('Clusters','image',2,21,nothing)
cv2.createTrackbar('Depth Weight1','image',0,9,nothing) # no float permitted
cv2.createTrackbar('Depth Weight2','image',0,16,nothing)# no float permitted 
cv2.createTrackbar('Coord Weight','image',0,1,nothing)
cv2.createTrackbar('Depth ThresUp','image',0,255,nothing)
cv2.createTrackbar('Depth ThresDown','image',0,255,nothing)

# Set the default values fot the trackbars
cv2.setTrackbarPos('Clusters','image', doc["clustering"]["number_of_clusters"])
cv2.setTrackbarPos('Depth Weight1','image', 2)
cv2.setTrackbarPos('Depth Weight2','image', 12)
cv2.setTrackbarPos('Coord Weight','image', doc["clustering"]["coordinates_weight"])
cv2.setTrackbarPos('Depth ThresUp','image', doc["clustering"]["depth_thresup"])
cv2.setTrackbarPos('Depth ThresDown','image', doc["clustering"]["depth_thresdown"])

# Create switch for ON/OFF functionality which indicates Start/Stop
switch = '0 : OFF \n 1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
  cv2.imshow('image',img)
  k = cv2.waitKey(1) & 0xFF
  if k == 27:
    break

  # Get current positions of trackbars
  nclusters = cv2.getTrackbarPos('Clusters','image')
  depth_weightn = cv2.getTrackbarPos('Depth Weight1','image')
  depth_weightp = cv2.getTrackbarPos('Depth Weight2','image')
  depth_weight = depth_weightn ** (-depth_weightp)
  coord_weight = cv2.getTrackbarPos('Coord Weight','image')
  depth_thresup = cv2.getTrackbarPos('Depth ThresUp','image')
  depth_thresdown = cv2.getTrackbarPos('Depth ThresDown','image')
  s = cv2.getTrackbarPos(switch,'image')

  if s == 1:
    img = clusterer.clusterer(rgbname,depthname,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown)
    img = metaproccessor.metaproccessor(clusteredname,rgbname,depthname,nclusters)
    cv2.setTrackbarPos(switch,'image', 0)
  #~ else:

cv2.destroyAllWindows()
