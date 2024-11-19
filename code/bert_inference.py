import pandas as pd
import numpy as np
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

test = pd.read_pickle('test_set.pkl').dropna()
train = pd.read_pickle('train_set.pkl').dropna()
test_text = Dataset.from_pandas(test)
labels = train['label'].values
labels= np.unique(labels)
del train
model_path = "thesis/BERT_finetuned2"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
bert= pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
pred=[]
for out in bert(KeyDataset(test_text, "sentence"), batch_size=32, truncation="only_first"):
    pred.append(out['label'])

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels = [label2id[lbl] for lbl in test['label'].values]
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
