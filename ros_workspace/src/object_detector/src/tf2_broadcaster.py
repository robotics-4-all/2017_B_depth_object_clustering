#!/usr/bin/env python 
# Based on http://wiki.ros.org/tf2/Tutorials/Adding%20a%20frame%20%28Python%29
import rospy
import tf2_msgs.msg
import geometry_msgs.msg
from object_detector.msg import Detected_object
from image_capturer import DetectedObject


class TFBroadcaster:
    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
        self.sub_det_obj = rospy.Subscriber("/object_found", Detected_object, self.object_callback)
        self.detected_objects = []
        self.counter_of_detected_objects = 0
      
    def object_callback(self, msg):
        self.counter_of_detected_objects += 1
        self.detected_objects.append(DetectedObject(msg.nameid, msg.x, msg.y, msg.z, msg.width, msg.height))
        print(len(self.detected_objects))

    def publish_tfs(self, event):
        for obj in self.detected_objects:
            #~ rospy.sleep(0.1)
            t = geometry_msgs.msg.TransformStamped()
            # It is the same Header of the messages of pointcloud.
            t.header.frame_id = "camera_rgb_optical_frame"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "Object" + str(obj.nameid)
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

    rospy.Timer(rospy.Duration(1), tfb.publish_tfs)
    rospy.spin()
