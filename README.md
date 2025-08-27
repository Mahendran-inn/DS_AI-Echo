# 💡 AI Echo: Your Smartest Conversational Partner 
Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a given text. This project aims to analyze user reviews of a ChatGPT application and classify them as positive, neutral, or negative based on the sentiment expressed. The goal is to gain insights into customersatisfaction,identify common concerns, and enhance the application's user experience. 

---


## 🚀 Features

- 📦 Cleaned and transformed data
- 🔍 EDA using Matplotlib, Seaborn
- 📁 Dataset divided for optimized model training (ML, DL, NLP with Streamlit)

---


## 🔍 Problem Statement
Predict whether a user is giving review of a ChatGPT application and classify them as positive, neutral, or negative based on the sentiment expressed

---

🖼️ Sample EDA Visuals:

<img width="531" height="393" alt="245b8e93-dad1-4b5e-903d-d3d501d80b4a" src="https://github.com/user-attachments/assets/8b1e0ef5-a77f-4846-a91c-2f90fd38fff4" />

<img width="856" height="486" alt="90817a1e-c6a4-4222-a50a-75215807ea1f" src="https://github.com/user-attachments/assets/118541c4-7789-4b89-966f-0e46d6b2fbfd" />



##  Machine Learning Model Training Script

```python

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

x = df_ml.drop(['label','review','date'], axis=1)
y = df_ml['label']

from sklearn.model_selection import train_test_split, GridSearchCV
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


lr=LogisticRegression()
knn=KNeighborsClassifier()

# Logistic Regression pipeline
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

lr_para = {
    'lr__fit_intercept': [True, False],
    'lr__max_iter': [200, 500, 1000],
    'lr__solver': ['lbfgs', 'saga']
}

# KNN pipeline
pipe_knn = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier())
])

knn_para = {
    'knn__n_neighbors': [3, 5, 7, 10],
    'knn__weights': ['uniform', 'distance']
}


# Logistic Regression
grid_lr = GridSearchCV(pipe_lr, lr_para, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(xtrain, ytrain)

# KNN
grid_knn = GridSearchCV(pipe_knn, knn_para, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(xtrain, ytrain)


# Best params and scores
print("Best Logistic Regression:", grid_lr.best_params_, "Score:", grid_lr.best_score_)
print("Best KNN:", grid_knn.best_params_, "Score:", grid_knn.best_score_)


```


##  Deep Learning Model Training Script

```python

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 1. Features & Labels
X_dl = df_ml.drop(['label','review','date'], axis=1).values   # features
y_dl = df_ml['label'].values  # labels (Negative/Neutral/Positive)

# Encode labels into integers
le_dl = LabelEncoder()
y_encoded = le_dl.fit_transform(y_dl)   # e.g: Negative=0, Neutral=1, Positive=2

# Convert to one-hot (needed for softmax output layer)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_categorical, test_size=0.2, random_state=42)

# Scale features
scaler_dl = StandardScaler()
X_train_dl = scaler_dl.fit_transform(X_train_dl)
X_test_dl = scaler_dl.transform(X_test_dl)


# 2. Deep Learning Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_dl.shape[1], activation='softmax')
])


# 3. Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    X_train_dl, y_train_dl, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=1)


# 4. Evaluate
loss, acc = model.evaluate(X_test_dl, y_test_dl, verbose=0)
# print(f"DL Test Accuracy: {acc:.4f}")

```



