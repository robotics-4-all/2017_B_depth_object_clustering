#!/usr/bin/env python
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import clusterer
import metaproccessor

class image_capturer:
  
  def __init__(self):
    self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.Rgbcallback)
    self.dpth_sub = rospy.Subscriber("/camera/depth_registered/image", Image, self.Depthcallback)
    self.bridge = CvBridge()
    self.desired_shape = (252, 336)
    self.rgbimg = np.ndarray(shape=self.desired_shape)
    self.depthimg = np.ndarray(shape=self.desired_shape)
  
  def Rgbcallback(self,msg_rgb):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="passthrough")
      # Resize to the desired size
      cv_image_resized = cv2.resize(cv_image, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      self.rgbimg = cv_image_resized
    except CvBridgeError as e:
      print(e)
    
  def Depthcallback(self,msg_depth): # TODO still too noisy!
    try:
      # The depth image is a single-channel float32 image
      # the values is the distance in mm in z axis
      cv_image = self.bridge.imgmsg_to_cv2(msg_depth, "passthrough")
      # Convert the depth image to a Numpy array since most cv2 functions
      # require Numpy arrays.
      cv_image_array = np.array(cv_image, dtype=np.float32)
      # Normalize the depth image to fall between 0 (black) and 1 (white)
      # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
      cv_image = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
      # Resize to the desired size
      cv_image_resized = cv2.resize(cv_image, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      self.depthimg = cv_image_resized
    except CvBridgeError as e:
      print(e)
  
  def Processor(self):
    [clusteredimg,img] = clusterer.clusterer(self.rgbimg,self.depthimg,6,0.04,0,70,0)
    cv2.imshow("Clustered Image", clusteredimg)
    cv2.waitKey(3)
  
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
