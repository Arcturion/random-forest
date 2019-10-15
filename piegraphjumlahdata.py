import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import export_graphviz
import graphviz
from subprocess import call
#from sklearn import tree

from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('datamalamedit.csv', sep=',')

data_array = df.values
X = data_array[:, 1:26]
y = data_array[:, 26]

y = y.astype(int)

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ambilnama = df.drop(columns = 'tanggal')
ambilnama = ambilnama.drop(columns = 'ww')
feature_names = list(ambilnama.columns)

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

X_train = X_train.astype(float)
X_test = X_test.astype(int)
y_train = y_train.astype(float)
y_test = y_test.astype(int)
start = time.time()
#, 32, 64, 128, 256, 512, 1024
parameters = {'n_estimators': [2, 4, 32, 64, 128, 256, 512, 1024], #banyaknya tree
              'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'max_depth': [4, 10, 20], #dalamnya tree
              'bootstrap': [True, False],}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='accuracy')
print_grid_search(clf, parameters)

#proses train
clf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
clf.fit(X_train, y_train)

#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

importances_index_desc = np.argsort(clf.feature_importances_)[::-1]
feature_labels = [feature_names[-i] for i in importances_index_desc]

plt.figure()
plt.bar(feature_labels, clf.feature_importances_[importances_index_desc])
plt.xticks(feature_labels, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Features')

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn,fp,fn,tp)

acc = accuracy_score(y_test, y_pred) * 100
print("Akurasi pengujian terhadap data uji: %.2f" % acc, "%")

end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

plt.show()
#for feature in zip(feat_labels, clf.feature_importances_):
 #   print(feature)
