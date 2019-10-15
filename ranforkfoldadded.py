import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
#from sklearn import tree

df = pd.read_csv('datamalam.csv', sep=',')

data_array = df.values
X = data_array[:, 1:23]
y = data_array[:, 23]

cv = KFold(n_splits=2)

feature_names = list(df.columns)

def print_grid_search(classifier, tuned_paramaters, scores=['accuracy']):
    for score in scores:
        #train                
        classifier.fit(X_train, y_train)
        print("Parameter terbaik:\n")        
        print(classifier.best_params_)
        print()
        print("Dengan akurasi:\n")        
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) untuk %r"
                  % (mean, std * 2, params))


#X_train = X_train.astype(float)
#X_test = X_test.astype(float)
#y_train = y_train.astype(float)
#y_test = y_test.astype(float)
 
X = X.astype(float)
y = y.astype(float)


parameters = {'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
              'criterion': ['gini', 'entropy']}
#clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, scoring='accuracy')
#print_grid_search(clf, parameters)

clf = RandomForestClassifier(n_estimators=256, criterion='entropy')
count =1

for train, test in cv.split(X, y):
    proba = clf.fit(X[train, :], y[train]).predict_proba(X[test])
    importances_index_desc = np.argsort(clf.feature_importances_)[::-1]
    feature_labels = [feature_names[-i] for i in importances_index_desc]

    plt.figure()
    plt.bar(feature_labels, clf.feature_importances_[importances_index_desc])
    plt.xticks(feature_labels, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.title('Fold {}'.format(count))
    count = count + 1
plt.show()
#proses train
#y_pred = clf.predict(X_test)

#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#print(tn,fp,fn,tp)


#for feature in zip(feat_labels, clf.feature_importances_):
 #   print(feature)
