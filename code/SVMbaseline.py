from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import class_weight
from sklearnex import patch_sklearn
patch_sklearn()
import joblib


df = pd.read_pickle("train_set.pkl")
df = df.fillna('')
X = df['sentence'].values
X_clean = []
for i in range(len(X)):
      X_clean.append(clean(X[i]))

y = df['label']


#SVM baseline
max_df = 1
min_df = 1
c_val = 0.55
voc = load_vacabulary()
vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, vocabulary=voc)
X = vectorizer.transform(X_clean)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#RBF kernel
#n_estimators = 1
#clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True), n_jobs=-1)
clf = svm.LinearSVC()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

labels = np.unique(y)
label2id = {label:idx for idx, label in enumerate(labels)}
labels = [label2id[lbl] for lbl in y_test]
pred = [label2id[lbl] for lbl in pred]
metrics={"accuracy": accuracy_score(y_true=labels, y_pred=pred), 
      "MSE": mean_squared_error(y_true=labels, y_pred=pred),
      "micro_precision": precision_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_precision": precision_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_precision": precision_score(y_true=labels, y_pred=pred, average='weighted'), 
      "micro_recall": recall_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_recall": recall_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_recall": recall_score(y_true=labels, y_pred=pred, average='weighted'), 
      "micro_f1": f1_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_f1": f1_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_f1": f1_score(y_true=labels, y_pred=pred, average='weighted') 
}
print(metrics)

joblib.dump(clf, "svm_baseline.pkl") 


#Accuracy: 15.69%
#F1 Score (micro): 15.69%
#F1 Score (macro): 0.19%
