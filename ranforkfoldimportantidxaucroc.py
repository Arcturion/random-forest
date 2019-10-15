import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
#from sklearn import tree

from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('datapagitanpa2019.csv', sep=',')

data_array = df.values
X = data_array[:, 1:40]
y = data_array[:, 40]

cv = KFold(n_splits=5)

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
#clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='accuracy')
#print_grid_search(clf, parameters)

clf = RandomForestClassifier()
count =1

#for train, test in cv.split(X, y):
#    clf.fit(X[train, :], y[train])
#    importances_index_desc = np.argsort(clf.feature_importances_)[::-1]
#    feature_labels = [feature_names[-i] for i in importances_index_desc]
#
#    plt.figure()
#    plt.bar(feature_labels, clf.feature_importances_[importances_index_desc])
#    plt.xticks(feature_labels, rotation='vertical')
#    plt.ylabel('Importance')
#    plt.xlabel('Features')
#    plt.title("Parameter Penting")
#   #plt.title('Fold {}'.format(count))
#    count = count + 1
#plt.show()

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

ros = RandomOverSampler(random_state=42)

mean_akur=0
mean_sensi=0
mean_spesi=0
i=0
print("tn, fp, fn, tp")
for train, test in cv.split(X,y):
    X_res, y_res = ros.fit_sample(X[train, :], y[train])
    clf.fit(X_res, y_res)
    probas = clf.predict_proba(X[test])
    y_pred = clf.predict(X[test])
    tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
    akur=(tn+tp)/(tn+fp+fn+tp)
    print(akur)
    mean_akur = mean_akur+akur
    fpr, tpr, threshold = roc_curve(y[test], probas[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

print()
print(mean_akur/5)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.legend([mean_auc])
plt.title('ROC AUC PAGI')
plt.ylabel('Sensitifitas')
plt.xlabel('Spesifisitas')
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.show()
#proses train
#y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn,fp)
print(fn,tp)


#for feature in zip(feat_labels, clf.feature_importances_):
 #   print(feature)
