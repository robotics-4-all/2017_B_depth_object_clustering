import cv2
import numpy as np
import time
import yaml

import clusterer
import metaprocessor


def nothing(x):
    pass
    return x


def gui_editor(rgb_img, depth_img):

    # Read the parameters from the yaml file
    with open("../cfg/conf.yaml", 'r') as stream:
        try:
            doc = yaml.load(stream)
            # Create a window with initial RGB and Depth images
            img = np.concatenate((rgb_img, cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)), axis=1)
            cv2.namedWindow('image')

            # Create track bars for parameters of clusterer
            cv2.createTrackbar('Clusters', 'image', 2, 21, nothing)
            cv2.createTrackbar('Depth Weight1', 'image', 0, 20, nothing)  # no float permitted
            cv2.createTrackbar('Depth Weight2', 'image', 0, 16, nothing)  # no float permitted
            # ~ cv2.createTrackbar('Coord Weight','image',0,1,nothing)
            cv2.createTrackbar('Depth ThreshUp', 'image', 0, 255, nothing)
            cv2.createTrackbar('Depth ThreshDown', 'image', 0, 255, nothing)

            # Set the default values fot the track bars
            cv2.setTrackbarPos('Clusters', 'image', doc["clustering"]["number_of_clusters"])
            cv2.setTrackbarPos('Depth Weight1', 'image', 3)
            cv2.setTrackbarPos('Depth Weight2', 'image', 1)
            # ~ cv2.setTrackbarPos('Coord Weight','image', doc["clustering"]["coordinates_weight"])
            cv2.setTrackbarPos('Depth ThreshUp', 'image', doc["clustering"]["depth_thresup"])
            cv2.setTrackbarPos('Depth ThreshDown', 'image', doc["clustering"]["depth_thresdown"])

            print("Press ENTER to start or Esc to exit.")
            bounding_boxes = []
            while 1:
                cv2.imshow('image', img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                if k == 13:
                    # Get current positions of track bars
                    n_clusters = cv2.getTrackbarPos('Clusters', 'image')
                    depth_weight_digit = cv2.getTrackbarPos('Depth Weight1', 'image')
                    depth_weight_power = cv2.getTrackbarPos('Depth Weight2', 'image')
                    depth_weight = depth_weight_digit * 10 ** (-depth_weight_power)
                    coord_weight = 0
                    depth_thresh_up = cv2.getTrackbarPos('Depth ThreshUp', 'image')
                    depth_thresh_down = cv2.getTrackbarPos('Depth ThreshDown', 'image')

                    # Apply the processing functions
                    start_time = time.time()
                    [clustered_img, _] = clusterer.clusterer(rgb_img, depth_img, n_clusters, depth_weight,
                                                             coord_weight, depth_thresh_up, depth_thresh_down)
                    [img, bounding_boxes] = metaprocessor.meta_processor(clustered_img, rgb_img, depth_img, n_clusters)
                    elapsed_time = time.time() - start_time
                    print("Object detection is done in time: " + str(elapsed_time) + "s!")
                    print("Press ENTER to start or Esc to exit.")
            return bounding_boxes
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    # Read the parameters from the yaml file
    with open("../cfg/conf.yaml", 'r') as Stream:
        try:
            Doc = yaml.load(Stream)
            Rgb_name = Doc["images"]["rgbname"]
            Depth_name = Doc["images"]["depthname"]
            Rgb_img = cv2.imread(Rgb_name)
            Depth_img = cv2.imread(Depth_name, cv2.IMREAD_GRAYSCALE)
            Depth_img_view = cv2.imread(Depth_name)
            gui_editor(Rgb_img, Depth_img)
            cv2.destroyAllWindows()
        except yaml.YAMLError as Exc:
            print(Exc)
