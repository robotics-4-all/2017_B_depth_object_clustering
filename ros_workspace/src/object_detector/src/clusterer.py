#!/usr/bin/python
# Filename: clusterer.py

from __future__ import division
import time
import numpy as np
import cv2
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def separate_objects_from_floor_and_wall(feature_vector, depth_weight, depth_thresh_up, depth_thresh_down):
    height, width, channels = feature_vector[:,:,0:3].shape
    feature_vector_array = feature_vector.reshape(height*width, 6)

    if depth_weight != 0:
        condition_up = feature_vector_array[:, -1] < depth_thresh_up
        condition_down = feature_vector_array[:, -1] > depth_thresh_down
        condition = np.logical_and(condition_up, condition_down)
        feature_vector_array = feature_vector_array[np.where(condition)]

    # Normalize features
    feature_vector_array = normalize(feature_vector_array, norm='max', axis=0)
    feature_vector_array[:, -1] = feature_vector_array[:, -1] * depth_weight

    k_means = KMeans(n_clusters=2).fit(feature_vector_array[:, [0, 1, 2, 5]])
    # TODO Luminosity removes also some white objects from white floor
    # TODO cannot put n_jobs = -1

    processed_img_depth = feature_vector[:, :, -1].copy()

    # Find the cluster that has the furthest points. Don't look thresholded ones.
    depth_sum_cluster0 = []
    depth_sum_cluster1 = []

    non_zero_depth_counter = 0
    for i in range(0, height):
        for j in range(0, width):
            if depth_thresh_down < feature_vector[i, j, -1] < depth_thresh_up:
                if k_means.labels_[non_zero_depth_counter] == 0:
                    depth_sum_cluster0.append(feature_vector[i, j, -1])
                else:
                    depth_sum_cluster1.append(feature_vector[i, j, -1])
                non_zero_depth_counter += 1

    # Find the average distance of each cluster.
    if sum(depth_sum_cluster0)/len(depth_sum_cluster0) > sum(depth_sum_cluster1)/len(depth_sum_cluster1):
        further_cluster = 0
    else:
        further_cluster = 1

    # Remove pixels that belong to the cluster that has the furthest points.
    non_zero_depth_counter = 0
    for i in range(0, height):
        for j in range(0, width):
            if depth_thresh_down < feature_vector[i, j, -1] < depth_thresh_up:
                if k_means.labels_[non_zero_depth_counter] == further_cluster:
                    processed_img_depth[i, j] = 0
                non_zero_depth_counter += 1
            else:
                processed_img_depth[i, j] = 0
    return processed_img_depth


def clusterer(img_rgb, img_depth, n_clusters, depth_weight, coord_weight, depth_thresh_up, depth_thresh_down):
    height, width, channels = img_rgb.shape

    # Convert the image from BGR to Lab color space
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2Lab)

    img_coord = np.zeros((height, width, 3), dtype=np.uint8)
    img_coord[:, :, 0] = np.transpose(np.tile(np.array(range(height)), (width, 1))) + 1
    img_coord[:, :, 1] = np.tile(np.array(range(width)), (height, 1)) + 1
    img_coord[:, :, 2] = img_depth[:, :]

    feature_vector = np.zeros((height, width, 6))
    feature_vector[:, :, 0:3] = img_lab
    feature_vector[:, :, 3:5] = img_coord[:, :, 0:1]  # x+y
    feature_vector[:, :, 5] = img_coord[:, :, 2]
    # TODO test in more depth the function separate_objects_from_floor_and_wall
    feature_vector[:, :, 5] = separate_objects_from_floor_and_wall(feature_vector, 4, 100, depth_thresh_down)  # z

    feature_vector_array = feature_vector.reshape(height*width, 6)

    # Remove noisy data based and threshold the whole frame on the depth camera
    if depth_weight != 0:
        condition_up = feature_vector_array[:, -1] < depth_thresh_up
        condition_down = feature_vector_array[:, -1] > depth_thresh_down
        condition = np.logical_and(condition_up, condition_down)
        feature_vector_array = feature_vector_array[np.where(condition)]

    # Normalize the features
    feature_vector_array = normalize(feature_vector_array, norm='max', axis=0)
    feature_vector_array[:, -1] = feature_vector_array[:, -1] * coord_weight

    start_time = time.time()
    # TODO check other methods of clustering
    k_means = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(feature_vector_array[:, [0, 1, 2, 5]])
    elapsed_time = time.time() - start_time
    print ("\nK-means with " + str(n_clusters) + " clusters is done with elapsed time " + str(elapsed_time) + "s!")
    segment_img = np.zeros((height, width, 3), dtype=np.uint8)

    with open("../cfg/conf.yaml", 'r') as stream:
        try:
            doc = yaml.load(stream)
            col_dict = doc["clustering"]["coldict"]
            non_zero_depth_counter = 0
            start_time = time.time()
            for i in range(0, height):
                for j in range(0, width):
                    if depth_thresh_up > feature_vector[i, j, -1] > depth_thresh_down:
                        segment_img[i, j, :] = col_dict[k_means.labels_[non_zero_depth_counter]]
                        non_zero_depth_counter += 1
                    else:
                        segment_img[i, j, :] = 0
            elapsed_time = time.time() - start_time
            print ("Labelling is done in time:" + str(elapsed_time) + "s!")

            vis = np.concatenate((img_rgb, cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR), segment_img), axis=1)
            name = "Results/result_" + str(n_clusters) + ".png"
            name_all = "Results/result_all_" + str(n_clusters) + ".png"
            cv2.imwrite(name, segment_img)
            cv2.imwrite('Results/result_cur.png', segment_img)
            cv2.imwrite(name_all, vis)
            return [segment_img, vis]
        except yaml.YAMLError as exc:
            print(exc)
    return 0


if __name__ == '__main__':
    # Read the parameters from the yaml file
    with open("../cfg/conf.yaml", 'r') as Stream:
        try:
            Doc = yaml.load(Stream)
            Rgb_name = Doc["images"]["rgbname"]
            Depth_name = Doc["images"]["depthname"]
            N_clusters = Doc["clustering"]["number_of_clusters"]  # TODO maybe find it from histogram of RGB
            Depth_weight = Doc["clustering"]["depth_weight"]
            Coord_weight = Doc["clustering"]["coordinates_weight"]
            Depth_thresh_up = Doc["clustering"]["depth_thresup"]
            Depth_thresh_down = Doc["clustering"]["depth_thresdown"]
            # Load the color image
            Img_rgb = cv2.imread(Rgb_name)
            # Load the depth image (aligned and grayscaled)
            Img_depth = cv2.imread(Depth_name, cv2.IMREAD_GRAYSCALE)

            [clustered, combined_img] = clusterer(Img_rgb, Img_depth, N_clusters, Depth_weight, Coord_weight,
                                                  Depth_thresh_up, Depth_thresh_down)
            cv2.imshow("Clustered Image", combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except yaml.YAMLError as Exc:
            print(Exc)
