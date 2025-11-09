import pandas as pd
import matplotlib.pyplot as plt

# data_path = "./data/spam.csv"
df = pd.read_csv(
    "./data/spam.csv",
    encoding="latin-1"
)

# conservar solo las columnas útiles    
df = df[['v1', 'v2']]
df.columns = ['etiqueta', 'mensaje']
df.head()
df.info()

print("=================================")

conteo_etiquetas = df['etiqueta'].value_counts()
print(conteo_etiquetas)

print("=================================")

conteo_etiquetas.plot(kind='bar',color=['green','red'])
plt.title("Conteo de mensajes por tipo")
plt.xlabel("Etiquetas")
plt.ylabel("Numero de mensajes")
plt.savefig("conteo_mensajes.png")

plt.show()

print(df.describe())

import numpy as np
from sklearn.model_selection import train_test_split

# # test_labels =
# np.loadtxt('dataset/test/test_labels.csv', delimiter=',')  # (50,)

# test_labels=np.loadtxt('spam.csv', delimiter=',') 

X = []

for i in range(df.shape[0]):
    X.append(df.iloc[i,1])

labels = []

for i in range(df.shape[0]):
    if df.iloc[i, 0] == "spam":
        labels.append(1)
    else:
        labels.append(0)

y = np.array(labels)

# print(x[:5]) 
# print(y[:5]) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
      test_size=0.30, 
      random_state=11, 
      stratify=y)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', min_df=1, lowercase=False)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("=================================")
print("=================================")

from sklearn.ensemble import RandomForestClassifier

print ('Training model.. ')
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=11,
    max_features="sqrt"
)

model.fit(X_train_vectorized, y_train)
print ('Saving model..')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test_vectorized)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=================================")
print("metrics")
print("=================================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f" F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")


import pathlib
from joblib import dump
dump(model, pathlib.Path('model/spamornot.joblib'))
dump(vectorizer, pathlib.Path('model/vectorizer.joblib'))
