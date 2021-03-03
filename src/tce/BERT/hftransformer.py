import string
import re
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from textwrap import wrap
from collections import defaultdict
from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import transformers


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00",
                        "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("drive/MyDrive/data/tceTextData.csv")
df.head()


def remove_urls(text):
    temp = re.sub("http.?://[^\s]+[\s]?", "", text)
    text = re.sub("(www\.)[^\s]+[\s]?", "", temp)
    return text


def remove_digits(text: str):
    return re.sub(r'\d+', '', text)


def remove_punctuation(text: str):
    punct = string.punctuation
    translation_table = str.maketrans(punct, len(punct)*' ')
    return text.translate(translation_table)


def clean_special_chars(text: str):
    return re.sub('[^A-Za-z0-9 $]+', '', text)


def preptext(text: str):
    return remove_urls(remove_digits(clean_special_chars(text.lower().capitalize())))


df.columns = ['empenho', 'natureza']

n_samples = int(df.empenho.values.shape[0] * 0.3)
X, y = resample(df.empenho, df.natureza, n_samples=n_samples, random_state=15)

X.shape, y.shape

X = X.apply(preptext)

df = np.dstack((X, y))[0]

df.shape

df = pd.DataFrame(data=df, columns=['empenho', 'natureza'])

lb = LabelEncoder()
df['encNatureza'] = lb.fit_transform(df.natureza)

len(lb.classes_)

PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []
for txt in df.empenho:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 512])
plt.xlabel('Token count')

MAX_LEN = 150


class TCEDataset(Dataset):
    def __init__(self, empenho, targets, tokenizer, max_len):
        self.empenho = empenho
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.empenho)

    def __getitem__(self, item):
        empenho = str(self.empenho[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            empenho,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'empenho_text': empenho,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=RANDOM_SEED
)

df_train.shape, df_val.shape, df_test.shape

df_train.info()


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TCEDataset(
        empenho=df.empenho.to_numpy(),
        targets=df.encNatureza.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


BATCH_SIZE = 16
train_data_loader = create_data_loader(
    df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


class NaturezaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(NaturezaClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(bert_output['pooler_output'])
        return self.out(output)


model = NaturezaClassifier(len(lb.classes_))
model = model.to(device)

#model.load_state_dict(torch.load('drive/MyDrive/data/best_model_state.bin', map_location=torch.device('cpu')))

EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        predictions.extend(preds)
        real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    macro = f1_score(real_values, predictions, average='macro')
    micro = f1_score(real_values, predictions, average='micro')
    return correct_predictions.double() / n_examples, np.mean(losses), macro, micro


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    macro = f1_score(real_values, predictions, average='macro')
    micro = f1_score(real_values, predictions, average='micro')
    return correct_predictions.double() / n_examples, np.mean(losses), macro, micro

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = defaultdict(list)
# best_accuracy = 0
# for epoch in range(EPOCHS):
#     print(f'Epoch {epoch + 1}/{EPOCHS}')
#     print('-' * 10)
#     train_acc, train_loss, train_macro, train_micro = train_epoch(
#             model,
#             train_data_loader,
#             loss_fn,
#             optimizer,
#             device,
#             scheduler,
#             len(df_train)
#         )
#     print(f'Train loss {train_loss} accuracy {train_acc} macro {train_macro} micro {train_micro}')
#     val_acc, val_loss, val_macro, val_micro = eval_model(
#             model,
#             val_data_loader,
#             loss_fn,
#             device,
#             len(df_val)
#         )
#     print(f'Val   loss {val_loss} accuracy {val_acc} macro {val_macro} micro {val_micro}')
#     print()
#     history['train_acc'].append(train_acc)
#     history['train_loss'].append(train_loss)
#     history['train_macro'].append(train_macro)
#     history['train_micro'].append(train_micro)
#     history['val_acc'].append(val_acc)
#     history['val_loss'].append(val_loss)
#     history['val_macro'].append(val_macro)
#     history['val_micro'].append(val_micro)
#     if val_acc > best_accuracy:
#         torch.save(model.state_dict(), 'drive/MyDrive/data/best_model_state.bin')
#         best_accuracy = val_acc


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='validation loss')
plt.title('Training history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.plot(history['train_macro'], label='train macro')
plt.plot(history['val_macro'], label='validation macro')
plt.title('Training history')
plt.ylabel('Macro')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.plot(history['train_micro'], label='train micro')
plt.plot(history['val_micro'], label='validation micro')
plt.title('Training history')
plt.ylabel('Micro')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)
test_acc.item()


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["empenho_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
)

class_names = lb.classes_

len(class_names)


print(classification_report(y_test, y_pred))


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(
        hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(
        hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

idx = 2
review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
    'class_names': class_names,
    'values': y_pred_probs[idx]
})

print("\n".join(wrap(review_text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')

sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1])

review_text = "I love completing my todos! Best app ever!!!"

encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction]}')


def predict_raw(review_text):
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print(f'Empenho text: {review_text}')
    print(f'Natureza pred: {class_names[prediction]}')


for n in range(0, 20):
    text = test_data_loader.dataset.__getitem__(n)['empenho_text']
    label = test_data_loader.dataset.__getitem__(n)['targets']
    predict_raw(text)
    print(f'Natureza real: {lb.inverse_transform([label.tolist()])}')

label.tolist()
