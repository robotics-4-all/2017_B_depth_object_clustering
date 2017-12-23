#!/usr/bin/env python  
import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
from object_detector.msg import Detected_object
from image_capturer import DetectedObject

class TFBroadcaster:
  def __init__(self):
    self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
    self.sub_det_obj = rospy.Subscriber("/object_found", Detected_object, self.Objectcallback)
    self.detected_objects = []
    self.counter_of_detected_objects = 0

    #~ while not rospy.is_shutdown():
      #~ # Run this loop at about 10Hz
      #~ rospy.sleep(0.1)

      #~ t = geometry_msgs.msg.TransformStamped()
      #~ t.header.frame_id = "kinect"
      #~ t.header.stamp = rospy.Time.now()
      #~ t.child_frame_id = "object1"
      #~ t.transform.translation.x = 1.0
      #~ t.transform.translation.y = 2.0
      #~ t.transform.translation.z = 3.0

      #~ t.transform.rotation.x = 0.0
      #~ t.transform.rotation.y = 0.0
      #~ t.transform.rotation.z = 0.0
      #~ t.transform.rotation.w = 1.0

      #~ tfm = tf2_msgs.msg.TFMessage([t])
      #~ self.pub_tf.publish(tfm)
      
  def Objectcallback(self, msg):
    self.counter_of_detected_objects += 1
    self.detected_objects.append(DetectedObject(msg.nameid, msg.x, msg.y, msg.z, msg.width, msg.height))

  def PublishTFs(self, event):
    print len(self.detected_objects)
    for obj in self.detected_objects:
      #~ rospy.sleep(0.1)
      t = geometry_msgs.msg.TransformStamped()
      t.header.frame_id = "kinect"
      t.header.stamp = rospy.Time.now()
      t.child_frame_id = str(obj.nameid)
      t.transform.translation.x = obj.x
      t.transform.translation.y = obj.y
      t.transform.translation.z = obj.z

      t.transform.rotation.x = 0.0
      t.transform.rotation.y = 0.0
      t.transform.rotation.z = 0.0
      t.transform.rotation.w = 1.0

      tfm = tf2_msgs.msg.TFMessage([t])
      self.pub_tf.publish(tfm)
    
if __name__ == '__main__':
  rospy.init_node('tf2_broadcaster')
  tfb = TFBroadcaster()
  
  rospy.Timer(rospy.Duration(0.1), tfb.PublishTFs)
  rospy.spin()
