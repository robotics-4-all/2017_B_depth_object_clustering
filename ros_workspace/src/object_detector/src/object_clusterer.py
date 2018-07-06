#!/usr/bin/env python
# Filename: object_clusterer.py
import rospy
import numpy as np
from sklearn.cluster import KMeans
from object_detector.msg import Point_feature_histogram
from sklearn.metrics import silhouette_score
from sklearn import preprocessing


class Objectclusterer:
    def __init__(self):
        self.sub_det_obj = rospy.Subscriber("/pfh_found", Point_feature_histogram, self.object_callback)
        self.feature_vector = []
        self.name_id_vector = []
        self.n_objects = 0

    def object_callback(self, msg):
        self.feature_vector.append(msg.pfh)
        self.name_id_vector.append(msg.name_id)
        temp = np.unique(self.name_id_vector, return_counts=1)

        # Number of different objects that have been found
        self.n_objects = len(temp[0])
        self.clusterer()

    def clusterer(self):
        min_max = preprocessing.MinMaxScaler(feature_range=(0, 1))
        norm_feature_vector = min_max.fit(self.feature_vector).transform(self.feature_vector)
        print len(self.feature_vector)
        for k in range(2, self.n_objects-1):
            k_means = KMeans(n_clusters=k, n_jobs=-1).fit(norm_feature_vector)
            cluster_labels = k_means.fit_predict(norm_feature_vector)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            silhouette_avg = silhouette_score(norm_feature_vector, cluster_labels)
            print("For n_clusters=" + str(k) + " The average silhouette_score is :" + str(silhouette_avg))
            print("For n_clusters=" + str(k) + str(cluster_labels) + str(self.name_id_vector))


if __name__ == '__main__':
    rospy.init_node('object_clusterer')
    clusterer = Objectclusterer()
    rospy.loginfo("object_clusterer node is up!")

    rospy.spin()
