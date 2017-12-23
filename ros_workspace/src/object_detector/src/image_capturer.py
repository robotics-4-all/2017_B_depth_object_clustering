#!/usr/bin/env python
import sys
import rospy
import cv2
import yaml
import numpy as np
import time
from sensor_msgs.msg import Image
from object_detector.msg import Detected_object
from cv_bridge import CvBridge, CvBridgeError
import gui_editor

class DetectedObject:
  def __init__(self, nameid, x, y, z, width, height):
    self.nameid = nameid
    self.x = x
    self.y = y
    self.z = z
    self.width = width
    self.height = height
    self.mu = (x,y,z)
    self.sigma = 1
  
  def __str__(self):
    string_to_print = 'Oject' + str(self.nameid) + ':(x:' + str(self.x) + ',y:' + str(self.y) + ',z:' + str(self.z) + ', width:' + str(self.width) +  ',height:' + str(self.height) + ')'
    return string_to_print
    
  def update_gaussdist(self, sigma, mu):
    self.mu = mu
    self.sigma = sigma

class image_capturer:  
  def __init__(self):
    self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.Rgbcallback)
    self.dpth_sub = rospy.Subscriber("/camera/depth_registered/image", Image, self.Depthcallback)
    self.obje_pub = rospy.Publisher('/object_found', Detected_object, queue_size=10)
    self.bridge = CvBridge()
    self.desired_shape = (336, 252)
    self.rgbimg = np.ndarray(shape=self.desired_shape, dtype = np.uint8)
    self.depthimg = np.ndarray(shape=self.desired_shape, dtype = np.uint8)
    self.detected_objects = []
    print "\nPress R if you want to trigger GUI for object detection..."
    print "Press Esc if you want to end the suffer of this node...\n"
    
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
      try:
        img = np.concatenate((self.rgbimg, cv2.cvtColor(self.depthimg,cv2.COLOR_GRAY2RGB)), axis=1)
        cv2.imshow("Combined image from my node", img)
      except ValueError as valerr:
        print "Images from channels are not ready yet..."
      k = cv2.waitKey(1) & 0xFF
      if k == 114: # if you press r, trigger the procressing
        self.Process()
        print "\nPress R if you want to trigger GUI for object detection..."
        print "Press Esc if you want to end the suffer of this node...\n"
      if k == 27: # if you press Esc kill the node
        rospy.signal_shutdown("Whatever")
    except CvBridgeError as e:
      print(e)

  def Depthcallback(self,msg_depth): # TODO still too noisy!
    try:
      # The depth image is a single-channel float32 image
      # the values is the distance in mm in z axis
      cv_image = self.bridge.imgmsg_to_cv2(msg_depth, "32FC1")
      # Convert the depth image to a Numpy array since most cv2 functions
      # require Numpy arrays.
      cv_image_array = np.array(cv_image, dtype = np.float64)
      # Normalize the depth image to fall between 0 (black) and 1 (white) in ordet to view result
      # Normalize the depth image to fall between 0 (black) and 255 (white) in ordet to write result
      # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
      cv_image_norm_write = cv_image_array.copy()
      cv_image_norm_write = cv2.normalize(cv_image_array, cv_image_norm_write, 0, 255, cv2.NORM_MINMAX)
      # Resize to the desired size
      cv_image_resized_write = cv2.resize(cv_image_norm_write, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      self.depthimg = cv_image_resized_write.astype(np.uint8)
    except CvBridgeError as e:
      print(e)
  
  def UpdatheworldCallback(self):
    for det_object in self.detected_objects:
      msg = Detected_object()
      msg.nameid = det_object.nameid
      msg.x = det_object.x
      msg.y = det_object.y
      msg.z = det_object.z
      msg.width = det_object.width
      msg.height = det_object.height
      self.obje_pub.publish(msg)
  
  def Process(self):
    bounding_boxes = gui_editor.gui_editor(self.rgbimg, self.depthimg)
    counter = len(self.detected_objects)
    for c in bounding_boxes:
      x, y, w, h = cv2.boundingRect(c)
      # TODO identify the same objects and update them
      z = self.depthimg[y+h/2][x+w/2] * 0.01
      centerx = (self.desired_shape[0]/2 - (x+w/2) ) * 0.01
      centery = (self.desired_shape[1] - (y+h/2) ) * 0.01
      print centerx,centery,z
      self.detected_objects.append(DetectedObject(counter, centerx, z, centery, w, h))
      counter += 1
    self.UpdatheworldCallback()
    
  
def main(args):
  ic = image_capturer()
  rospy.init_node('image_capturer', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
