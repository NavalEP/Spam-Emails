# Spam-Emails
 
Welcome to the Spam Email/SMS Detection project! This repository contains a machine learning pipeline to detect spam messages (emails or SMS) using text preprocessing and classification techniques. The goal is to accurately classify messages as either 'spam' or 'ham' (not spam).


Table of Contents:
1 Introduction
2 Data
3 Installation
4 Usage
5 Model Training and Evaluation
6 Results
7 Saving the Model

Introduction
Spam detection is a critical task in modern communication systems. This project uses a dataset of labeled messages to build a machine learning model that can differentiate between spam and ham messages. The process involves data cleaning, text preprocessing, feature extraction, model training, and evaluation.

Data
The dataset used in this project is spam.csv, which contains the following columns:

v1: Label indicating whether the message is 'spam' or 'ham'.
v2: The message content.
Unnamed columns which are removed during preprocessing.
Installation
To get started with this project, follow these steps:

Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Data Loading and Preprocessing
Load the dataset and preprocess the data:
python
Copy code
import pandas as pd

df = pd.read_csv('path/to/spam.csv', encoding='latin1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'output', 'v2': 'sms/email'}, inplace=True)
df['output'] = df['output'].map({'ham': 0, 'spam': 1})
Text Processing
Define a function to preprocess the text:
python
Copy code
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

def process_txt(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word.isalnum()]
    txt = [word for word in txt if word not in stopwords.words('english') and word not in string.punctuation]
    ps = PorterStemmer()
    txt = [ps.stem(word) for word in txt]
    return " ".join(txt)

df['text_transform'] = df['sms/email'].apply(process_txt)
Feature Extraction
Extract features using TF-IDF:
python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['text_transform']).toarray()
y = df['output'].values
Train-Test Split
Split the data into training and testing sets:
python
Copy code
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)
Model Training and Evaluation
Train a Naive Bayes model:

python
Copy code
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

Evaluate the model:

python
Copy code
print(accuracy_score(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))

Visualize the confusion matrix:

python
Copy code
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_gnb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
Results
The model's accuracy, confusion matrix, and classification report provide insights into its performance.
The confusion matrix visualization helps understand the distribution of true and false positives/negatives.
Saving the Model


Save the trained model and vectorizer:
python
Copy code
import pickle

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(gnb, open('model.pkl', 'wb'))
