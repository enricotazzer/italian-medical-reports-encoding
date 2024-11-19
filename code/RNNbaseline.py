import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, SpatialDropout1D
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error

df = pd.read_pickle("train_set.pkl")
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
Y = pd.get_dummies(df['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SimpleRNN(100))
model.add(Dense(500, activation='relu'))
model.add(Dense(2606, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 5
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
model.save('RNNbaseline.keras')

y_pred = model.predict(X_test)
pred = np.argmax(y_pred, axis=1)
labels = np.nonzero(Y_test==True)[1]

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
