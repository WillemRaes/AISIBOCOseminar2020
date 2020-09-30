import os
import sys
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.io.arff import loadarff

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.neighbors import NearestNeighbors as KNN
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
import seaborn

path = os.getcwd()
print(path)

# http datasets WITHOUT duplicates and with idf weighted categorical attributes. Not normalized
raw_data = loadarff('KDDCup99_withoutdupl_idf.arff')
raw_data_list = 'KDDCup99_withoutdupl_idf.arff'
df = pd.DataFrame(raw_data[0])
print(df['outlier'].unique())
normal = [b'no']
anomaly = [b'yes']

# Set labels in dataframe 0 : inlier 1: outlier
normal_dict  = {x: 0 for x in normal}
anomaly_dict = {x: 1 for x in anomaly}
label_dict = {**normal_dict, **anomaly_dict}
print(label_dict)
df.replace({"outlier": label_dict}, inplace=True)

# Define random state for reproducibility
random_state = np.random.RandomState(42)

# Define dataframes
df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
              'IForest']
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
ap_df = pd.DataFrame(columns=df_columns)
prc_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)


df = df.reset_index()
X = df.loc[:, df.columns != 'label']
y = df['outlier'].ravel()
outliers_fraction = np.sum(y == 1) / len(y)
outliers_percentage = round(outliers_fraction * 100, ndigits=4)


# 60% data for training and 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=random_state) # shuffle=True by default

# standardizing data for processing
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

# Create a dictionary of possible model candidates
classifiers = {
    'IsolationForest': IsolationForest(contamination=outliers_fraction, random_state=random_state)
    # 'KNN': KNN(n_neighbors=85)

}

# Unsupervised model training and preformance evaluation (in this case only 1 classifier is considered)
for clf_name, clf in classifiers.items():
    t0 = time()

    # train model
    clf.fit(X_train_norm)

    # Sklearn metrics compatibility
    test_scores = clf.decision_function(X_test_norm) * -1
    # test_scores = clf.score_samples(X_test_norm) *-1
    precision, recall, thresholds = precision_recall_curve(y_test, test_scores)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    y_pred = clf.predict(X_test_norm)

    # Isolation forest returns -1 for outlier and 1 for inlier (fix for sklearn metrics, change attribute pos_label
    # also possible)
    y_pred = [1 if val == -1 else 0 for val in y_pred]
    t1 = time()
    duration = round(t1 - t0, ndigits=4)

    # ROC metric focuses on both classes and is not suited for performance evaluation on imbalanced datasets
    roc = roc_auc_score(y_test, test_scores)
    print('Isolation forest unsupervised ROC AUC: %.3f' % roc)

    # Precision score: performance on the outlier class TP / (TP + FP)
    prn = precision_score(y_test, y_pred)
    print('Isolation forest unsupervised Precision: %.3f' % prn)

    # Precision Area Under Curve score: Good performance parameter for severe imbalanced datasets
    prc = auc(recall, precision)
    print('Isolation forest unsupervised PR AUC: %.3f' % prc)

    # Plot the Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    seaborn.heatmap(conf_mat, annot=True, cbar=False, cmap="YlGnBu", linewidths=.5, fmt="d")
    plt.show()

    # Convert into ONNX format
    initial_type = [('float_input', FloatTensorType([None, 43]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open(clf_name + ".onnx", "wb") as f:
        f.write(onx.SerializeToString())


