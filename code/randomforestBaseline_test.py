from utils import *
from preprocessing import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
patch_sklearn()
import joblib

df = pd.read_pickle("test_set.pkl")
df = df.fillna('')
X = df['sentence'].values
X_clean = []
for i in range(len(X)):
	X_clean.append(clean(X[i]))

MAX_NB_WORDS = 3000
MAX_SEQUENCE_LENGTH = 165
EMBEDDING_DIM = 165
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_clean)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(X_clean)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

rf = joblib.load("rf_baseline.pkl")

y = df['label'].values
label = pd.read_pickle('train_set.pkl')['label'].values
label2id = {label:idx for idx, label in enumerate(np.unique(label))}
del label
labels = [label2id[lbl] for lbl in y]

pred = rf.predict(X)
pred=[label2id[lbl] for lbl in pred]
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
print({"accuracy": accuracy_score(y_true=labels, y_pred=pred), 
      "MSE": mean_squared_error(y_true=labels, y_pred=pred),
      "micro_precision": precision_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_precision": precision_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_precision": precision_score(y_true=labels, y_pred=pred, average='weighted'), 
      "micro_recall": recall_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_recall": recall_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_recall": recall_score(y_true=labels, y_pred=pred, average='weighted'), 
      "micro_f1": f1_score(y_true=labels, y_pred=pred, average='micro'), 
      "macro_f1": f1_score(y_true=labels, y_pred=pred, average='macro'), 
      "weighted_f1": f1_score(y_true=labels, y_pred=pred, average='weighted'), 
})
