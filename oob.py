import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import export_graphviz
import graphviz
from subprocess import call
#from sklearn import tree

df = pd.read_csv('datapagiedittanpa2019.csv', sep=',')

data_array = df.values
X = data_array[:, 1:26]
y = data_array[:, 26]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ambilnama = df.drop(columns = 'tanggal')
ambilnama = ambilnama.drop(columns = 'ww')
feature_names = list(ambilnama.columns)

X = X_train.astype(float)
X_test = X_test.astype(float)
y = y_train.astype(float)
y_test = y_test.astype(float)
start = time.time()

RANDOM_STATE = 43

ensemble_clfs = [

    ("max_features='sqrt'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features='sqrt',
                               criterion='gini',
                               max_depth=30,
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("max_features='log2'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features='log2',
                               criterion='gini',
                               max_depth=30,
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("max_features=None",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features=None,
                               criterion='gini',
                               max_depth=30,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 20
max_estimators = 2000

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
