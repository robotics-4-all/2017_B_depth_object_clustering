#!/usr/bin/env python
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_capturer:
  
  def __init__(self):
    self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.Rgbcallback)
    self.dpth_sub = rospy.Subscriber("/camera/depth_registered/image", Image, self.Depthcallback)
    self.bridge = CvBridge()
    self.desired_shape = (336, 252)
  
  def Rgbcallback(self,msg_rgb):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="passthrough")
      cv_image_resized = cv2.resize(cv_image, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      #~ cv2.imshow("RGB Image", cv_image_resized)
      #~ cv2.waitKey(3)
    except CvBridgeError as e:
      print(e)
    
  def Depthcallback(self,msg_depth):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
      cv_image_resized = cv2.resize(cv_image, self.desired_shape, interpolation = cv2.INTER_CUBIC)
      #~ cv_image1 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      #~ cvuint8 = cv2.convertScaleAbs(cv_image1)
      print cv_image_resized
      cv2.imshow("Depth Image", cv_image_resized)
      cv2.waitKey(3)
    except CvBridgeError as e:
      print(e)
  
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
