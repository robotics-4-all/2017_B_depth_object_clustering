#!/usr/bin/env python
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

filename = 'Database/pfh.csv'
df = pandas.read_csv(filename, delimiter=',')
# drop by Name
df = df.drop(['Name'], axis=1)

# Split the set into testing and validating sets.
validation_size = 0.20
seed = 7
X = df.loc[:, df.columns != 'Id']
Y = df.loc[:, df.columns == 'Id'].values.ravel()

X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = [('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('SVM', SVC())]
# evaluate each model in turn based on scoring
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    k_fold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print("\nSpecifically for SVM model:")
print("Accuracy is " + str(accuracy_score(Y_validation, predictions)))
print("Confusion Matrix is \n" + str(confusion_matrix(Y_validation, predictions)))
print("Classification report is\n " + str(classification_report(Y_validation, predictions)))