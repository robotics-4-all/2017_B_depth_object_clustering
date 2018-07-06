#!/usr/bin/env python
# Filename: object_detector_node.py
from __future__ import print_function
import rospy
import tf
from tf import TransformListener
import cv2
import yaml
import numpy as np

from sklearn.cluster import KMeans
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2

from object_detector.msg import Detected_object, Point_feature_histogram
from object_detector.srv import Box
from cv_bridge import CvBridge, CvBridgeError
import gui_editor


def print_cielab_without_opencv(color):
    l = color[0] * 100 / 255
    a = color[1] - 128
    b = color[2] - 128
    print(l, a, b)


# Returns the point in PointCloud regarding to the pixel in image.
def return_pcl(x_img, y_img, pcl):
    # Solution maybe to find the median of the bounding box.
    if (y_img >= pcl.height) or (x_img >= pcl.width):
        raise NameError('Something is wrong with the translations between image and Pointcloud.')
    data_out = list(pc2.read_points(pcl, field_names=("x", "y", "z"), skip_nans=True, uvs=[[x_img, y_img]]))
    int_data = data_out[0]
    return int_data


class DetectedObject:
    def __init__(self, name_id, pcl_x, pcl_y, pcl_z, world_x, world_y, world_z, width, height,
                 crop_rgb_img=None, crop_depth_img=None, pointcloud=None):
        self.name_id = name_id
        self.width = width  # real dimension of object in x-axis
        self.height = height  # real dimension of object in y-axis
        self.length = 0  # real dimension of object in z-axis

        self.x = world_x  # center of the object
        self.y = world_y  # center of the object
        self.z = world_z  # center of the object

        self.pcl_x = pcl_x
        self.pcl_y = pcl_y
        self.pcl_z = pcl_z

        self.sigma_x = self.width / 6  # standard deviation in x-axis
        self.sigma_y = self.height / 6  # standard deviation in y-axis

        self.crop_rgb_img = crop_rgb_img  # TODO make it array in order to have better visualizations of diff frames.
        self.crop_depth_img = crop_depth_img  # TODO make it array in order to have better visualizations of diff frames
        self.whole_pointcloud = pointcloud
        self.pfh = []  # list of pfhs of the object
        self.pfh.append(self.crop_pointcloud_client())
        if crop_rgb_img is not None:
            # Find the two dominants colors of the detected object using k-means with two clusters.
            height_img, width_img, channels = crop_rgb_img.shape
            # Convert Image from BGR to Cie-Lab to compute color distance.
            image = cv2.cvtColor(crop_rgb_img, cv2.COLOR_BGR2LAB)
            image_array = image.reshape(height_img * width_img, channels)
            num_dom_colors = 2
            k_means = KMeans(n_clusters=num_dom_colors, init='random', n_jobs=1).fit(image_array)
            # Grab the number of different clusters & create a histogram based on the number of pixels
            # assigned to each cluster. After that, find the cluster with most pixels - that will be the dominant color.
            num_labels = np.arange(0, len(np.unique(k_means.labels_)) + 1)
            (hist, _) = np.histogram(k_means.labels_, bins=num_labels)
            self.dom_colors = k_means.cluster_centers_[np.argmax(hist)].astype(np.uint8)

    def __str__(self):
        string_to_print = 'Object' + str(self.name_id) + ':(x:' + str(self.x) + ',y:' + str(self.y) + ',z:' \
                          + str(self.z) + ', width:' + str(self.width) + ',height:' + str(self.height) + ')'
        cv2.imwrite('Results/Object' + str(self.name_id) + '.png', self.crop_rgb_img)
        return string_to_print

    def is_the_same_object(self, other_object):
        # Color distance
        dist_color = np.sqrt((int(self.dom_colors[0]) - int(other_object.dom_colors[0])) ** 2 +
                             (int(self.dom_colors[1]) - int(other_object.dom_colors[1])) ** 2 +
                             (int(self.dom_colors[2]) - int(other_object.dom_colors[2])) ** 2)
        # L in [0, 100], a in [-127, 127], b in [-127, 127]
        # And according to OpenCV L=L*255/100,a=a+128,b=b+128 for 8-bit images
        # L in [0, 255], a in [1, 255], b in [1, 255]
        norm_dist_color = dist_color / np.sqrt(255 ** 2 + 254 ** 2 + 254 ** 2)

        # Real Width and Height distance
        dist_width = abs(self.width - other_object.width)
        norm_dist_width = dist_width / max(self.width, other_object.width)
        dist_height = abs(self.height - other_object.height)
        norm_dist_height = dist_height / max(self.height, other_object.height)

        # Use norm_function to get the probability to be the same object
        dist_prob = 1 - self.norm_function(other_object.x, other_object.y, other_object.z)
        weights = [10, 1, 1, 10]
        norm_distances = [norm_dist_color, norm_dist_width, norm_dist_height, dist_prob]
        dist_final = np.inner(norm_distances, weights)
        norm_dist_final = dist_final / sum(weights)
        norm_prob_final = 1 - norm_dist_final
        # print("Final normalised probability to be object-" + str(self.name_id + 1) + " same with object-" +
        #       str(other_object.name_id + 1) + ": " + str(norm_prob_final) + "\n")

        if norm_prob_final > 0.6:
            print("object-" + str(self.name_id + 1) + " is same with object-" +
                  str(other_object.name_id + 1) + ": " + str(norm_prob_final))
            return True
        else:
            return False

    def update_dimensions(self, newly_observed_object):
        # Update the width, height and length according to the new "frame" of the object.
        # Treat real dimensions and sigmas in a different way.
        # TODO think of what happens if you see less and less of the object. The dimensions are getting smaller.
        # TODO Maybe take this approach only when it gets bigger and bigger.
        self.width = (self.width + newly_observed_object.width) / 2
        self.height = (self.height + newly_observed_object.height) / 2
        # Average the two curves of normal distributions
        # Credits: https://stats.stackexchange.com/questions/179213/mean-of-two-normal-distributions
        # Credits: http://epaxon.blogspot.gr/2012/12/bayesian-inference-with-probabilistic.html
        self.sigma_x = np.sqrt((self.sigma_x ** 2 + newly_observed_object.width ** 2) / 2)
        self.sigma_y = np.sqrt((self.sigma_y ** 2 + newly_observed_object.height ** 2) / 2)
        sum_sigmas2_x = self.sigma_x ** 2 + newly_observed_object.sigma_x ** 2
        sum_sigmas2_y = self.sigma_y ** 2 + newly_observed_object.sigma_y ** 2
        self.x = self.x * (newly_observed_object.sigma_x ** 2 /
                           sum_sigmas2_x) + newly_observed_object.x * (self.sigma_x ** 2/sum_sigmas2_x)
        self.y = self.y * (newly_observed_object.sigma_y ** 2 /
                           sum_sigmas2_y) + newly_observed_object.y * (self.sigma_y ** 2 / sum_sigmas2_y)
        if self.length < abs(newly_observed_object.z - self.z):
            self.length = abs(newly_observed_object.z - self.z)

    def norm_function(self, x_inp, y_inp, z_inp):
        if self.x - self.sigma_x <= x_inp <= self.x + self.sigma_x:
            px = 0.682
        elif self.x - 3 * self.sigma_x < x_inp < self.x - self.sigma_x\
                or self.x + self.sigma_x < x_inp < self.x + 3 * self.sigma_x:
            px = 0.314
        else:
            px = 0.004

        if self.y - self.sigma_y <= y_inp <= self.y + self.sigma_y:
            py = 0.682
        elif self.y - 3 * self.sigma_y < y_inp < self.y - self.sigma_y\
                or self.y + self.sigma_y < y_inp < self.y + 3 * self.sigma_y:
            py = 0.314
        else:
            py = 0.004

        # If you have never noticed the object again, you have no information about the length in z-axis.
        # TODO needs testing...
        # The dimensions are in meters. So assume that objects with same characteristics are at least 1.5 meters far
        # away from each other.
        # TODO use self.length
        if self.z - z_inp < 1.5:
            pz = 1
        else:
            pz = 0
        return px * py * pz

    def crop_pointcloud_client(self):
        rospy.wait_for_service('crop_pointcloud')
        try:
            crop_pointcloud = rospy.ServiceProxy('crop_pointcloud', Box)
            object_pointcloud = crop_pointcloud(self.pcl_x, self.pcl_y, self.pcl_z, self.width,
                                                self.height, self.whole_pointcloud)  # TODO replace z with median z?
            return object_pointcloud.pfh
        except rospy.ServiceException as e:
            print("Service call failed: " + str(e) + "\n")


class ObjectDetector:
    def __init__(self):
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.rgb_callback)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pointcloud_callback)
        self.depth_raw_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.raw_depth_callback)
        # TODO subscribe only once
        self.camera_info_sub = rospy.Subscriber("camera/rgb/camera_info", CameraInfo, self.camera_info_callback)

        self.object_pub = rospy.Publisher('/object_found', Detected_object, queue_size=10)
        self.pfh_pub = rospy.Publisher('/pfh_found', Point_feature_histogram, queue_size=10)

        self.tf = TransformListener()
        self.stop_callbacks_flag = False
        # Holds the time that the instance has been taken.
        self.current_time = rospy.Time(0)

        # Camera infos - they will be updated with the Callback, hopefully.
        self.cx_d = 0
        self.cy_d = 0
        self.fx_d = 0
        self.fy_d = 0

        self.bridge = CvBridge()
        initial_shape = (480, 640)
        # Divide it by a number, to scale the image and make computations faster.
        self.desired_divide_factor = 2
        self.desired_shape = map(lambda x: x / self.desired_divide_factor, initial_shape)

        self.rgb_img = np.ndarray(shape=self.desired_shape, dtype=np.uint8)
        self.depth_img = np.ndarray(shape=self.desired_shape, dtype=np.uint8)
        self.depth_raw_img = np.ndarray(shape=self.desired_shape, dtype=np.uint8)
        self.pcl = PointCloud2()
        # Stores the overall objects that have been found.
        self.detected_objects = []
        # Stores the objects that have been found in the current instance.
        self.newly_detected_objects = []
        print("\nPress R if you want to trigger GUI for object detection...")
        print("Press Esc if you want to end the suffer of this node...\n")

        # Read the parameters from the yaml file
        with open("../cfg/conf.yaml", 'r') as stream:
            try:
                doc = yaml.load(stream)
                self.depth_weight = doc["clustering"]["depth_weight"]
                self.coordinates_weight = doc["clustering"]["coordinates_weight"]
                self.n_clusters = doc["clustering"]["number_of_clusters"]
                self.depth_thresh_up = doc["clustering"]["depth_thresup"]
                self.depth_thresh_down = doc["clustering"]["depth_thresdown"]
            except yaml.YAMLError as exc:
                print(exc)

    def camera_info_callback(self, msg_info):
        self.cx_d = msg_info.K[2]
        self.cy_d = msg_info.K[5]
        self.fx_d = msg_info.K[0]
        self.fy_d = msg_info.K[4]

    def rgb_callback(self, msg_rgb):
        try:
            # TODO check if it is also ok with kinect
            cv_image = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="bgr8")
            # Resize to the desired size
            cv_image_resized = cv2.resize(cv_image, tuple(reversed(self.desired_shape)), interpolation=cv2.INTER_AREA)
            self.rgb_img = cv_image_resized
            try:
                img = np.concatenate((self.rgb_img, cv2.cvtColor(self.depth_img, cv2.COLOR_GRAY2BGR)), axis=1)
                cv2.imshow("Combined image from my node", img)
            except ValueError as error:
                print(error)
                print("Images from channels are not ready yet...")
            k = cv2.waitKey(1) & 0xFF
            if k == 114:  # if you press r, trigger the processing
                self.stop_callbacks_flag = True
                self.current_time = rospy.Time(0)
                cv2.destroyAllWindows()
                try:
                    self.process()
                except NameError as warn:
                    cv2.destroyAllWindows()
                    print(warn)
                self.stop_callbacks_flag = False
                print("\nPress R if you want to trigger GUI for object detection...")
                print("Press Esc if you want to end the suffer of this node...\n")
            if k == 27:  # if you press Esc, kill the node
                rospy.signal_shutdown("Whatever...")
        except CvBridgeError as e:
            print(e)

    def raw_depth_callback(self, msg_raw_depth):
        # Raw image from device. Contains uint32 depths in mm.
        temp_img = self.bridge.imgmsg_to_cv2(msg_raw_depth, "passthrough")
        self.depth_raw_img = np.array(self.desired_shape, dtype=np.uint32)
        self.depth_raw_img = cv2.resize(temp_img, tuple(reversed(self.desired_shape)), interpolation=cv2.INTER_NEAREST)
        nans_positions = np.isnan(self.depth_raw_img)
        self.depth_raw_img[nans_positions] = 0
        self.depth_raw_img = cv2.convertScaleAbs(self.depth_raw_img, alpha=(255.0 / np.amax(self.depth_raw_img)))
        self.depth_img = self.depth_raw_img

    def pointcloud_callback(self, msg_pcl):
        # synchronize pcl with images
        if not self.stop_callbacks_flag:
            self.pcl = msg_pcl

    def process(self):
        bounding_boxes = gui_editor.gui_editor(self.rgb_img, self.depth_img, len(self.detected_objects))
        counter = len(self.detected_objects) + 1
        # For every newly found object.
        for c in bounding_boxes:
            x, y, w, h = cv2.boundingRect(c)
            # Take the center of the bounding box of the object.
            center_x = x + w / 2
            center_y = y + h / 2
            # Get the point from point cloud of the corresponding pixel.
            # Multiply by the desired factor the pixel's position, because I have scaled the images by this number.
            try:
                coords = return_pcl(center_x * self.desired_divide_factor,
                                    center_y * self.desired_divide_factor,
                                    self.pcl)
            except (NameError, IndexError):
                rospy.logwarn("WARNING: Found object" + str(counter) + " with gaps, let's ignore it!")
                continue
            # Based on formula: x3D = (x * 2 - self.cx_d) * z3D/self.fx_d
            # Based on formula: y3D = (y * 2 - self.cy_d) * z3D/self.fy_d
            real_width = self.desired_divide_factor * w * coords[2] / self.fx_d
            real_height = self.desired_divide_factor * h * coords[2] / self.fy_d

            # Crop the image and get just the bounding box.
            crop_rgb_img = self.rgb_img[y:y + h, x:x + w]
            crop_depth_img = self.depth_img[y:y + h, x:x + w]

            # Find the position of detected object relative to world.
            try:
                # Credits: http://www.euclideanspace.com/maths/geometry/affine/matrix4x4/
                [translation, rotation] = self.tf.lookupTransform('map', 'camera_rgb_optical_frame', self.current_time)
                transformation_matrix = self.tf.fromTranslationRotation(translation, rotation)
                old_point_vector = np.asanyarray([coords[0], coords[1], coords[2], 1])
                world_coordinates = transformation_matrix.dot(old_point_vector)
                [world_x, world_y, world_z] = world_coordinates[0:3]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # In case of not turtlebot and just kinect.
                rospy.logwarn("Cannot use transformation to the the frame of the world.")
                world_x, world_y, world_z = coords[0], coords[1], coords[2]
            det_object = DetectedObject(counter, coords[0], coords[1], coords[2],
                                        world_x, world_y, world_z, real_width, real_height, crop_rgb_img,
                                        crop_depth_img, self.pcl)
            self.newly_detected_objects.append(det_object)
            counter += 1
        self.update_world()

    def update_world(self):
        detected_objects_length = len(self.detected_objects)
        # For every new object that was found, first check whether it exists and then send it to the tf2_broadcaster.
        for new_det_object in self.newly_detected_objects:
            # Create the message for object_clusterer with the Point Feature Histogram and leave name_id for later.
            new_pfh_msg = Point_feature_histogram()
            new_pfh_msg.pfh = new_det_object.pfh[0]
            # First time with no already found objects.
            if detected_objects_length == 0:
                self.save_and_send(new_det_object)
                # Name_id is the same name_id with the one of the the object because it is the first time you see it.
                new_pfh_msg.name_id = new_det_object.name_id
                self.pfh_pub.publish(new_pfh_msg)
                continue
            for i in range(0, detected_objects_length):
                if self.detected_objects[i].is_the_same_object(new_det_object):
                    # It is the same object - so you have to update the dimensions  of it.
                    self.detected_objects[i].update_dimensions(new_det_object)
                    # Save another pfh of the current frame of the same object.
                    self.detected_objects[i].pfh.append(new_det_object.crop_pointcloud_client())
                    # Name_id is the same with the one of the already_found object as long as they are the same.
                    new_pfh_msg.name_id = self.detected_objects[i].name_id
                    self.pfh_pub.publish(new_pfh_msg)
                    break
                # If you have checked the newly found object with all already found ones, save and send it.
                if i == detected_objects_length - 1:
                    self.save_and_send(new_det_object)
                    # Name_id is the same name_id with the one of the the object because it is the first time you see it
                    new_pfh_msg.name_id = new_det_object.name_id
                    self.pfh_pub.publish(new_pfh_msg)
        # Empty the list newly_detected_objects
        del self.newly_detected_objects[:]
        print("Objects so far found: " + str(len(self.detected_objects)))

    # Saves the newly found object in the list of detected objects and sends it to tf2_broadcaster node.
    def save_and_send(self, to_send_object):
        print(to_send_object)
        self.detected_objects.append(to_send_object)
        msg = Detected_object()
        msg.name_id = to_send_object.name_id
        msg.x = to_send_object.x
        msg.y = to_send_object.y
        msg.z = to_send_object.z
        msg.width = to_send_object.width
        msg.height = to_send_object.height
        self.object_pub.publish(msg)


def main():
    ObjectDetector()
    rospy.init_node('object_detector_node', anonymous=True)
    rospy.loginfo("object_detector_node is up!")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
