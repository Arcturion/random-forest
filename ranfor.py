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

df = pd.read_csv('datapagitanpa2019.csv', sep=',')
df2 = pd.read_csv('datapagi2019.csv', sep=',')

data_array = df.values
data_array2 = df2.values
X_train = data_array[:, 1:40]
y_train = data_array[:, 40]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)

X_test =  data_array2[:, 1:40]
y_test = data_array2[:, 40]


ros = RandomOverSampler(random_state=42)


ambilnama = df.drop(columns = 'tanggal')
ambilnama = ambilnama.drop(columns = 'ww')
feature_names = list(ambilnama.columns)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

X_res, y_res = ros.fit_sample(X_train, y_train)

X_res = X_res.astype(float)
y_res = y_res.astype(float)

def print_grid_search(classifier, tuned_paramaters, scores=['accuracy','precision','recall','f1','roc_auc']):
    for score in scores:
        #train                
        classifier.fit(X_res, y_res)
        print(score)
        print("Parameter terbaik:\n")        
        print(classifier.best_params_)
        print()
        print("Dengan akurasi:\n")        
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) untuk %r"
                  % (mean, std * 2, params))


#, 32, 64, 128, 256, 512, 1024
parameters = {'n_estimators' : [1050],
                'criterion': ['gini'],
              'max_features': ['sqrt'],
              'max_depth': [30], #dalamnya tree
              'bootstrap': [True, False],}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='accuracy')
start = time.time()
#print_grid_search(clf, parameters)

#proses train
clf = RandomForestClassifier(n_estimators=2050, criterion='gini', max_depth=50, max_features='sqrt', bootstrap=False)
clf.fit(X_res, y_res)

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
plt.title('PARAMETER PENTING (MALAM)')
plt.ylabel('Importance')
plt.xlabel('Features')

y_pred = clf.predict(X_test)

#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#print(tn,fp)
#print(fn,tp)

acc = accuracy_score(y_test, y_pred) * 100
print("Akurasi pengujian terhadap data uji: %.2f" % acc, "%")

end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

plt.show()
#for feature in zip(feat_labels, clf.feature_importances_):
 #   print(feature)
