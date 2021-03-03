# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tce.data.sampling import get_topN_labels_doc
from tce.data.textpreprocessing import tfidfPprocessing
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
SEED = 15
# %%
df = pd.read_csv('/home/daniel/PIBIC2/data/tceTextData.csv')
df.columns = ['empenho', 'natureza']
df = get_topN_labels_doc(df, 'natureza', 400)

# %%
n_samples = int(df.values.shape[0] * 0.33)
newdf = resample(df, n_samples=n_samples,
                 random_state=SEED, stratify=df.natureza)
# %%
newdf.empenho = newdf.empenho.apply(tfidfPprocessing)
# %%
tfv = TfidfVectorizer()
X = tfv.fit_transform(newdf.empenho)
le = LabelEncoder()
y = le.fit_transform(newdf.natureza)
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, stratify=y_test, random_state=SEED)

# %%
clf = SVC(random_state=SEED)

# %%
clf.fit(X_train, y_train)
# %%
y_val_pred = clf.predict(X_val)
print('Validation Classification Report')
print(classification_report(y_val, y_val_pred))
# %%
y_test_pred = clf.predict(X_test)
print('Test Classification Report')
print(classification_report(y_test, y_test_pred))
