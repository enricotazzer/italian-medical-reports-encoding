from utils import *
from preprocessing import *
import numpy as np 
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from sklearnex import patch_sklearn
patch_sklearn()

df = pd.read_pickle("train_set.pkl").dropna()
df = df.fillna('')
X = df['sentence'].values
X_clean = []
for i in range(len(X)):
      X_clean.append(clean(X[i]))
MAX_NB_WORDS = 3000
MAX_SEQUENCE_LENGTH = 165
EMBEDDING_DIM = 165
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_clean)
X = tokenizer.texts_to_sequences(X_clean)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, "scaler_lr.scaler") 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LogisticRegression(multi_class='multinomial', warm_start=True, n_jobs=-1)
lr_model = lr_model.fit(X_train, y_train)
pred = lr_model.predict(X_test)

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

joblib.dump(lr_model, "lr_baseline.pkl") 

#0.26912181303116145
#Precision: 0.26912181303116145 / Recall: 0.26912181303116145
