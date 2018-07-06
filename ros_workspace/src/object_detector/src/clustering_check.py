#!/usr/bin/env python
import pandas
import numpy as np

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import itertools


filename = 'Database/pfh_6.csv'
df = pandas.read_csv(filename, delimiter=',')
# Choose how many objects you want to cluster.
df = df[df.Id < 7]

# drop by Name
df = df.drop(['Name'], axis=1)
X = df.loc[:, df.columns != 'Id']
Y = df.loc[:, df.columns == 'Id'].values.ravel()
# Find the number of objects in your dataset.
temp = np.unique(Y, return_counts=1)
n_labels = len(temp[0])

# Preprocess the features.
# Preprocessing Method 1
# axis used to normalize the data along. If 1, independently normalize each sample,
# otherwise (if 0) normalize each feature.
# X = preprocessing.normalize(X, norm='max', axis=0)
# Preprocessing Method 2
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
min_max = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = min_max.fit(X).transform(X)
# Preprocessing Method 3
# standard = preprocessing.StandardScaler().fit(X)
# X = standard.transform(X)

# Split the dataset into training and validation sets.
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = [('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier())]
# evaluate each model in turn based on scoring
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    k_fold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=scoring, n_jobs=-1)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset with k-nearest neighbors algorithm
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("\nSpecifically for kNN model:")
print("Accuracy for validation set is " + str(accuracy_score(Y_validation, predictions)))
print("Confusion Matrix is \n" + str(confusion_matrix(Y_validation, predictions)))
print("Classification report is\n " + str(classification_report(Y_validation, predictions)))

# Check if silhouette works
max_k = 0
max_silhouette_avg = 0
# TODO check other methods of clustering
k_means = KMeans(n_clusters=n_labels, n_jobs=-1, max_iter=500, n_init=20).fit(X)
cluster_labels = k_means.fit_predict(X)

# Classes to Cluster evaluation
combinations = np.array(list(itertools.permutations(range(n_labels))))
max_f1_score = 0
for comb in combinations:
    new_cluster_labels = list(cluster_labels)
    for i in range(0, len(cluster_labels)):
        new_cluster_labels[i] = int(comb[cluster_labels[i]])
    if max_f1_score < f1_score(Y, new_cluster_labels, average="micro"):
        max_f1_score = f1_score(Y, new_cluster_labels, average="micro")
        saved_cluster_labels = list(new_cluster_labels)

print("\nFor Classes to Cluster evaluation:")
print("F1_score is " + str(f1_score(Y, saved_cluster_labels, average="micro")))
print("Confusion Matrix is \n" + str(confusion_matrix(Y, saved_cluster_labels)))
print("Clustering report is\n " + str(classification_report(Y, saved_cluster_labels)))

# for k in range(2, 50):
#     k_means = KMeans(n_clusters=k, n_jobs=-1, max_iter=500, n_init=20).fit(X)
#     cluster_labels = k_means.fit_predict(X)
#
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed clusters.
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     # Find the value of k, for which the silhouette is being maximized
#     if silhouette_avg > max_silhouette_avg and k > 3:
#         max_silhouette_avg = silhouette_avg
#         max_k = k
#     print("For n_clusters=" + str(k) + ", the average silhouette score is :" + str(silhouette_avg))
# print("\nExcluding the values k = 2 and k = 3 for comparison.")
# print("Best n_clusters is " + str(max_k) + " with average silhouette score: " + str(max_silhouette_avg) +
#       " while the real number of classes is " + str(n_labels) + ".\n")
