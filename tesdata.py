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
X_train = data_array[:, 1:40]
y_train = data_array[:, 40]

data_array2 = df2.values
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

start = time.time()
#print_grid_search(clf, parameters)

#proses train
clf = RandomForestClassifier(n_estimators=1050, criterion='gini', max_depth=30, max_features='sqrt', bootstrap=False)
clf.fit(X_res, y_res)
proba = clf.predict_proba(X_test)

importances_index_desc = np.argsort(clf.feature_importances_)[::-1]
feature_labels = [feature_names[-i] for i in importances_index_desc]

plt.figure()
plt.bar(feature_labels, clf.feature_importances_[importances_index_desc])
plt.xticks(feature_labels, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.title('Variable Penting Malam')
plt.show()

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn,fp)
print(fn,tp)

acc = accuracy_score(y_test, y_pred) * 100
print("Akurasi pengujian terhadap data uji: %.2f" % acc, "%")

end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

eaea = pd.DataFrame(proba[:,1])
eaea = eaea.rename(columns={0:'prob'})
pred = pd.DataFrame(y_pred)
pred = pred.rename(columns={0:'y_pred'})
tesss = pd.DataFrame(y_test)
tesss= tesss.rename(columns={0:'y_test'})
param = pd.DataFrame(X_test)

param = pd.concat([param, tesss], axis=1)
param = pd.concat([param, pred], axis=1)
param = pd.concat([param, eaea], axis=1)

param.to_csv('parampagi.csv', sep=',', index=False)
#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
