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

def separate_objects_from_floorandwall(imglab, imgcoord, depth_weight, depth_thresup, depth_thresdown):
  height, width, channels = imglab.shape
  feature_vector = np.zeros((height,width,6))
  feature_vector[:,:,0:3] = imglab
  feature_vector[:,:,3:5] = imgcoord[:,:,0:1] * 0 # x+y
  
  depth_weight = 0.04
  feature_vector[:,:,5] = imgcoord[:,:,2] * depth_weight # z
  feature_vectorarray = feature_vector.reshape(height*width,6)

  if depth_weight != 0:
    conditionup = feature_vectorarray[:,-1] < depth_thresup * depth_weight
    conditiondown = feature_vectorarray[:,-1] > depth_thresdown * depth_weight
    condition = np.logical_and(conditionup, conditiondown)
    feature_vectorarray = feature_vectorarray[np.where(condition)]
  
  kmeans = KMeans(n_clusters=2 ,n_jobs=-1).fit(feature_vectorarray[:,[0,1,2,5]]) # TODO Luminosity removes also some white objects from white floor
  
  processed_imgdepth = imgcoord[:,:,2].copy()
  # Find the cluster that has the furthest points. Don't look thresholded ones.
  depth_sum_cluster0 = []
  depth_sum_cluster1 = []
  
  nonzerosdepth_counter = 0
  for i in range(0, height):
    for j in range(0, width):
      if imgcoord[i,j,2] > depth_thresdown and imgcoord[i,j,2] < depth_thresup:
        if kmeans.labels_[nonzerosdepth_counter] == 0:
          depth_sum_cluster0.append(imgcoord[i,j,2])
        else:
          depth_sum_cluster1.append(imgcoord[i,j,2])
        nonzerosdepth_counter += 1
  
  # Find the average distance of each cluster.
  if sum(depth_sum_cluster0)/len(depth_sum_cluster0) > sum(depth_sum_cluster1)/len(depth_sum_cluster1):
    further_cluster = 0
  else:
    further_cluster = 1
    
  # Remove pixels that belong to the cluster that has the furthest points.
  nonzerosdepth_counter = 0
  for i in range(0, height):
    for j in range(0, width):
      if imgcoord[i,j,2] > depth_thresdown and imgcoord[i,j,2] < depth_thresup:
        if kmeans.labels_[nonzerosdepth_counter] == further_cluster:
          processed_imgdepth[i,j] = 0
        nonzerosdepth_counter += 1
      else:
        processed_imgdepth[i,j] = 0
  return processed_imgdepth


def clusterer(imgrgb,imgdepth,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown):

  height, width, channels = imgrgb.shape

  # Convert the image to Lab color space 
  imglab = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2Lab)

  L = imglab[:,:,0]
  a = imglab[:,:,1]
  b = imglab[:,:,2]
  
  # TODO check it, maybe it's useless
  # Scaling
  sigma_L = np.std(L)
  sigma_a = np.std(a)
  sigma_b = np.std(b)
  imglab = (3/(sigma_L+sigma_a+sigma_b)) * imglab

  imgcoord = np.zeros((height,width,3), dtype = np.uint8)
  imgcoord[:,:,0] = np.transpose(np.tile(np.array(range(height)),(width,1))) + 1
  imgcoord[:,:,1] = np.tile(np.array(range(width)),(height,1)) + 1
  imgcoord[:,:,2] = imgdepth[:,:]


  # TODO maybe they are useless - check it
  sigma_x = np.std(np.std(imgcoord[:,:,0])) + 0.0000000001 # avoid zeros
  sigma_y = np.std(np.std(imgcoord[:,:,1])) + 0.0000000001 # avoid zeros
  sigma_z = np.std(np.std(imgcoord[:,:,2])) + 0.0000000001 # avoid zeros

  #~ imgcoord[:,:,0:2] = (2/(sigma_x + sigma_y)) * imgcoord[:,:,0:2]
  #~ imgcoord[:,:,2] = (1/(sigma_z)) * imgcoord[:,:,2]
  
  feature_vector = np.zeros((height,width,6))
  feature_vector[:,:,0:3] = imglab
  feature_vector[:,:,3:5] = imgcoord[:,:,0:1] * coord_weight # x+y
  feature_vector[:,:,5] = separate_objects_from_floorandwall(imglab, imgcoord, depth_weight, 150, depth_thresdown) * depth_weight # z

  feature_vectorarray = feature_vector.reshape(height*width,6)
  
  # Remove noisy data based on the depth camera
  if depth_weight != 0:
    conditionup = feature_vectorarray[:,-1] < depth_thresup * depth_weight
    conditiondown = feature_vectorarray[:,-1] > depth_thresdown * depth_weight
    condition = np.logical_and(conditionup, conditiondown)
    feature_vectorarray = feature_vectorarray[np.where(condition)]
  
  start_time = time.time()
  # TODO check other methods of clustering
  kmeans = KMeans(n_clusters=nclusters,n_jobs=-1).fit(feature_vectorarray[:,[0,1,2,5]])
  elapsed_time = time.time() - start_time
  print "\nKmeans with", nclusters, "clusters is done with elapsed time", elapsed_time, "s!"
  segmimg = np.zeros((height,width,3),dtype=np.uint8)

  with open("../cfg/conf.yaml", 'r') as stream:
    try:
      doc = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  coldict = doc["clustering"]["coldict"]

  nonzerosdepth_counter = 0
  start_time = time.time()
  for i in range(0, height):
    for j in range(0, width):
      if feature_vector[i,j,5] > depth_thresdown * depth_weight and  feature_vector[i,j,5] < depth_thresup * depth_weight:
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
  return [segmimg,vis]

if __name__ == '__main__':
  
  # Read the parameters from the yaml file
  with open("../cfg/conf.yaml", 'r') as stream:
    try:
      doc = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)
    
  rgbname = doc["images"]["rgbname"]
  depthname = doc["images"]["depthname"]
  nclusters = doc["clustering"]["number_of_clusters"] # TODO maybe find it from histogram of RGB
  depth_weight = doc["clustering"]["depth_weight"]
  coord_weight = doc["clustering"]["coordinates_weight"]
  depth_thresup = doc["clustering"]["depth_thresup"]
  depth_thresdown = doc["clustering"]["depth_thresdown"]
  # Load the color image
  imgrgb = cv2.imread(rgbname)
  # Load the depth image (aligned and grayscaled)
  imgdepth = cv2.imread(depthname,cv2.IMREAD_GRAYSCALE)
  
  [clustered,vis] = clusterer(imgrgb,imgdepth,nclusters,depth_weight,coord_weight,depth_thresup,depth_thresdown)
  cv2.imshow("Clustered Image", vis)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
