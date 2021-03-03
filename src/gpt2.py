import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
# from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

seed = 15
epochs = 4
batch_size = 16
max_length = 150
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'pierreguillou/gpt2-small-portuguese'


df = pd.read_csv('C:\\Users\\jefma\\GitHub\\pibic-cht\\data\\tceTextData.csv')
df.columns = ['empenho', 'natureza']
df = get_topN_labels_doc(df, 'natureza', 400)

n_samples = int(df.values.shape[0] * 0.33)
newdf = resample(df, n_samples=n_samples,
                 random_state=seed, stratify=df.natureza)
newdf.empenho = newdf.empenho.apply(embeddingPrep)

le = LabelEncoder()
df['encodedNatureza'] = le.fit_transform(df.natureza)
n_labels = len(le.classes_)


class TCEDataset(Dataset):
    def __init__(self, empenho, targets, tokenizer, max_len):
        self.empenho = empenho
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.empenho)

    def __getitem__(self, item):
        return {
            'text': str(self.empenho[item]),
            'label': self.targets[item]
        }


def create_data_loader(df, tokenizer, max_len, batch_size, collator):
    ds = TCEDataset(
        empenho=df.empenho.to_numpy(),
        targets=df.encodedNatureza.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator
    )


class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer, max_sequence_len=None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        return

    def __call__(self, sequences):

        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt",
                                    padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(dataloader, optimizer_, scheduler_, device_):

    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, device_):
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, config=model_config)

model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = model.config.eos_token_id

model.to(device)
print('Model loaded to `%s`' % device)

gpt2_classificaiton_collator = Gpt2ClassificationCollator(
    use_tokenizer=tokenizer, max_sequence_len=max_length)

df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    random_state=seed
)
df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=seed
)

train_dataloader = create_data_loader(
    df_train, tokenizer, max_length, batch_size)
valid_dataloader = create_data_loader(
    df_val, tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(
    df_test, tokenizer, max_length, batch_size)

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # default is 1e-8.
                  )

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}

print('Epoch')
for epoch in tqdm(range(epochs)):
    print()
    print('Training on batches...')
    train_labels, train_predict, train_loss = train(
        train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(train_labels, train_predict)

    print('Validation on batches...')
    valid_labels, valid_predict, val_loss = validation(
        valid_dataloader, device)
    val_acc = accuracy_score(valid_labels, valid_predict)

    print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" %
          (train_loss, val_loss, train_acc, val_acc))
    print()

    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)

true_labels, predictions_labels, avg_epoch_loss = validation(
    valid_dataloader, device)

evaluation_report = classification_report(true_labels, predictions_labels,)
print(evaluation_report)
