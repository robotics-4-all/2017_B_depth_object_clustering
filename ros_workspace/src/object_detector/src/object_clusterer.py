#!/usr/bin/env python
import rospy
import numpy as np
from sklearn.cluster import KMeans
from object_detector.msg import Point_feature_histogram
from sklearn.metrics import silhouette_score


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
        self.n_objects = len(temp[0])

    def clusterer(self):
        for k in range(2, self.n_objects):
            k_means = KMeans(n_clusters=k, n_jobs=-1).fit(self.feature_vector)
            cluster_labels = k_means.fit_predict(self.feature_vector)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            silhouette_avg = silhouette_score(self.feature_vector, cluster_labels)
            print("For n_clusters =" + str(k) + " The average silhouette_score is :" + str(silhouette_avg))


if __name__ == '__main__':
    rospy.init_node('object_clusterer')
    clusterer = Objectclusterer()
    rospy.spin()
