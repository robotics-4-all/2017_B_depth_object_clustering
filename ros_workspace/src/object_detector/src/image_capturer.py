#!/usr/bin/env python
import sys
import rospy
import cv2
import yaml
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import clusterer
import metaproccessor

class image_capturer:
  
  def __init__(self):
    self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.Rgbcallback)
    self.dpth_sub = rospy.Subscriber("/camera/depth_registered/image", Image, self.Depthcallback)
    self.bridge = CvBridge()
    self.desired_shape = (336, 252)
    self.rgbimg = np.ndarray(shape=self.desired_shape)
    self.depthimg = np.ndarray(shape=self.desired_shape)
    
    # Read the parameters from the yaml file
    with open("../cfg/conf.yaml", 'r') as stream:
      try:
        doc = yaml.load(stream)
        self.depth_weight = doc["clustering"]["depth_weight"]
        self.coordinates_weight = doc["clustering"]["coordinates_weight"]
        self.nclusters = doc["clustering"]["number_of_clusters"]
        self.depth_thresup = doc["clustering"]["depth_thresup"]
        self.depth_thresdown = doc["clustering"]["depth_thresdown"]
      except yaml.YAMLError as exc:
        print(exc) 
  
  def Rgbcallback(self,msg_rgb):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="passthrough")
      # Resize to the desired size
      cv_image_resized = cv2.resize(cv_image, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      self.rgbimg = cv_image_resized
      cv2.imwrite("Database/kinectrgb.png",self.rgbimg)
    except CvBridgeError as e:
      print(e)
    
  def Depthcallback(self,msg_depth): # TODO still too noisy!
    try:
      # The depth image is a single-channel float32 image
      # the values is the distance in mm in z axis
      cv_image = self.bridge.imgmsg_to_cv2(msg_depth, "32FC1")
      # Convert the depth image to a Numpy array since most cv2 functions
      # require Numpy arrays.
      cv_image_array = np.array(cv_image, dtype = float)
      # Normalize the depth image to fall between 0 (black) and 1 (white) in ordet to view result
      # Normalize the depth image to fall between 0 (black) and 255 (white) in ordet to write result
      # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
      cv_image_norm_write = cv_image_array.copy()
      cv_image_norm_show = cv_image_array.copy()
      cv_image_norm_write = cv2.normalize(cv_image_array, cv_image_norm_write, 0, 255, cv2.NORM_MINMAX)
      cv_image_norm_show = cv2.normalize(cv_image_array, cv_image_norm_show, 0, 1, cv2.NORM_MINMAX)
      # Resize to the desired size
      cv_image_resized_write = cv2.resize(cv_image_norm_write, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      cv_image_resized_show = cv2.resize(cv_image_norm_show, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      #~ cv_image_resized = cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2GRAY)
      self.depthimg = cv_image_resized_write
      cv2.imwrite("Database/kinectd.png",self.depthimg)
      cv2.imshow("Image from my node", cv_image_resized_show)
      cv2.waitKey(1)
    except CvBridgeError as e:
      print(e)
  
  def Process(self):
    [clusteredimg,img] = clusterer.clusterer(self.rgbimg,self.depthimg,self.nclusters,self.depth_weight,0,self.depth_thresup,self.depth_thresdown)
    img = metaproccessor.metaproccessor(clusteredimg,rgbimg,depthimg,self.nclusters,200)
    cv2.imshow("Clustered Image", img)
    cv2.waitKey(1)
  
def main(args):
  ic = image_capturer()
  rospy.init_node('image_capturer', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
