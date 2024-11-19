import torch 
import pandas as pd
import wandb
wandb.init(mode="disabled")
from huggingface_hub import login
login(token="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")
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

df = pd.read_pickle('train_set.pkl').dropna()
#df

labels = df['label'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label={id:label for id,label in enumerate(labels)}
label2id={label:id for id,label in enumerate(labels)}
df['label_num'] = df.label.map(lambda x: label2id[x.strip()])
#df
tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased', num_labels=num_labels, id2label=id2label, label2id=label2id)
model.to('cuda')

train_encoding = tokenizer(df['sentence'].tolist(), truncation=True, padding=True)

class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataloader = DataLoader(train_encoding, df['label_num'].tolist())

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
   
    # Extract true labels from the input object
    labels = pred.label_ids
    
    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)
    
    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)
    
    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='output',
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,                 
    train_dataset=train_dataloader,           
    compute_metrics= compute_metrics
)

trainer.train()

model_path = "thesis/BERT_finetuned2"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
