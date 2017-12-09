#!/usr/bin/python
# Filename: clusterer.py

from __future__ import division
import time
import sys
import math
import numpy as np
import cv2
import yaml
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def clusterer(rgbname,depthname,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown):

  # Load the color image
  imgrgb = cv2.imread(rgbname)
  # Load the depth image (aligned and grayscaled)
  imgdepth = cv2.imread(depthname,cv2.IMREAD_GRAYSCALE)
  # TODO divide image it into 2 or 3 sections with adaptive thresholding
  #~ hist = cv2.calcHist([imgdepth],[0],None,[256],[0,256])
  #~ plt.plot(hist[1:]) # ignore zeros
  #~ plt.xlim([1,256])
  #~ plt.show()

  height, width, channels = imgrgb.shape

  # Convert the image to Lab color space 
  imglab = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2Lab)

  L = imglab[:,:,0]
  a = imglab[:,:,1]
  b = imglab[:,:,2]

  # Scaling
  sigma_L = np.std(L)
  sigma_a = np.std(a)
  sigma_b = np.std(b)
  imglab = (3/(sigma_L+sigma_a+sigma_b)) * imglab

  imgcoord = np.zeros((height,width,3))
  imgcoord[:,:,0] = np.transpose(np.tile(np.array(range(height)),(width,1))) + 1
  imgcoord[:,:,1] = np.tile(np.array(range(width)),(height,1)) + 1
  imgcoord[:,:,2] = imgdepth[:,:]


  # TODO maybe they are useless - check it
  sigma_x = np.std(np.std(imgcoord[:,:,0])) + 0.0000000001 # avoid zeros
  sigma_y = np.std(np.std(imgcoord[:,:,1])) + 0.0000000001 # avoid zeros
  sigma_z = np.std(np.std(imgcoord[:,:,2])) + 0.0000000001 # avoid zeros

  imgcoord[:,:,0:2] = (2/(sigma_x + sigma_y)) * imgcoord[:,:,0:2]
  imgcoord[:,:,2] = (1/(sigma_z)) * imgcoord[:,:,2]

  feature_vector = np.zeros((height,width,6))
  feature_vector[:,:,0:3] = imglab
  feature_vector[:,:,3:5] = imgcoord[:,:,0:1] * coord_weight # x+y
  feature_vector[:,:,5] = imgcoord[:,:,2] * depth_weight # z

  feature_vectorarray = feature_vector.reshape(height*width,6)
  
  # Remove useless data based on the depth camera
  conditionup = feature_vectorarray[:,-1] < depth_thresup * (depth_weight/(sigma_z))
  conditiondown = feature_vectorarray[:,-1] > depth_thresdown * (depth_weight/(sigma_z))
  condition = np.logical_and(conditionup, conditiondown)
  feature_vectorarray = feature_vectorarray[np.where(condition)]
  
  start_time = time.time()
  # TODO check other methods of clustering
  kmeans = KMeans(n_clusters=nclusters,n_jobs=-1).fit(feature_vectorarray[:,1:]) # use only a and b
  elapsed_time = time.time() - start_time
  print "\nKmeans with", nclusters, "clusters is done with elapsed time", elapsed_time, "s!"
  segmimg = np.zeros((height,width,3),dtype=np.uint8)

  coldict = {
    0: [230, 25, 75],
    1: [60, 180, 75],
    2: [255, 225, 25],
    3: [0, 130, 200],
    4: [245, 130, 48],
    5: [145, 30, 180],
    6: [70, 240, 240],
    7: [240, 50, 230],
    8: [210, 245, 60],
    9: [250, 190, 190],
    10: [0, 128, 128],
    11: [230, 190, 255],
    12: [170, 110, 40],
    13: [255, 250, 200],
    14: [128, 0, 0],
    15: [170, 255, 195],
    16: [128, 128, 0],
    17: [255, 215, 180],
    18: [0, 0, 128],
    19: [128, 128, 128],
    20: [255, 255, 255],
    21: [0, 0, 0]
  }

  nonzerosdepth_counter = 0
  start_time = time.time()
  for i in range(0, height):
    for j in range(0, width):
      if imgdepth[i,j] > depth_thresdown and imgdepth[i,j] < depth_thresup: # apply a depth threshold
        segmimg[i,j,:] = coldict[kmeans.labels_[nonzerosdepth_counter]]
        nonzerosdepth_counter += 1
      else:
        segmimg[i,j,:] = 0
  elapsed_time = time.time() - start_time
  print "Labelling is done in time:", elapsed_time,"s!"
  vis = np.concatenate((imgrgb, cv2.cvtColor(imgdepth,cv2.COLOR_GRAY2RGB), segmimg), axis=1)
  name = "Results/result_" + str(nclusters) + ".png"
  nameall = "Results/resultall_" + str(nclusters) + ".png"
  cv2.imwrite(name,segmimg)
  cv2.imwrite('Results/result_cur.png',segmimg)
  cv2.imwrite(nameall,vis)
  return vis

if __name__ == '__main__':
  
  # Read the parameters from the yaml file
  with open("conf.yaml", 'r') as stream:
    try:
      doc = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
    
  rgbname = doc["images"]["rgbname"]
  depthname = doc["images"]["depthname"]
  nclusters = doc["clustering"]["number_of_clusters"] # TODO maybe find it from histogram of RGB
  depth_weight = doc["clustering"]["depth_weight"]
  coord_weight = doc["clustering"]["coordinates_weight"]
  depth_thresup = doc["clustering"]["depth_thresup"]  # TODO maybe find it from histogram of Depth
  depth_thresdown = doc["clustering"]["depth_thresdown"]
  vis = clusterer(rgbname,depthname,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown)
  cv2.imshow("Clustered Image", vis)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
