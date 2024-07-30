import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\naval\OneDrive\Desktop\Spam Emails\spam.csv', encoding='latin1')
df.head(10)

df.info()

#get cols names
df.columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)
df
df.rename(columns={'v1':'output','v2':'sms/email'},inplace=True)
df
#check Nulls
df.isnull().sum()

#check duplicates
num_duplicates = df.duplicated().sum()
num_duplicates, df.shape #shape before dropping

#drop duplicates
df = df.drop_duplicates(keep='first')
df.shape #shape after dropping 

import matplotlib.pyplot as plt


value_counts = df['output'].value_counts()
value_counts


plt.figure(figsize=(8, 6))
ax = value_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Distribution of Target col')
plt.xlabel('Labels')
plt.ylabel('Count')

#add num above each col
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

plt.show()

# Converting 'ham' to 0 and 'spam' to 1 (encoding)
df['output'] = df['output'].map({'ham': 0, 'spam': 1})
df['output']

import nltk
nltk.download('punkt')
nltk.download('stopwords')

#get num of characters in each sms
df['num_of_chars']=df['sms/email'].apply(len)
df

# Tokenize the 'sms' (get each word individually ),count these words
tokenized_sms = []
length_of_sms = []

for sms in df['sms/email']:
    tokens = nltk.word_tokenize(sms)
    tokenized_sms.append(tokens)
    length_of_sms.append(len(tokens))


df['tokenized_sms'] = tokenized_sms
df['num_of_words'] = length_of_sms

df.head()

from nltk.tokenize import sent_tokenize

#get num of sentence in each sms

num_sentences = []

for sms in df['sms/email']:
    sentences = sent_tokenize(sms)
    num_sentences.append(len(sentences))


df['num_of_sent'] = num_sentences

df.head()

#describe these 3 cols to get more info 

df[['num_of_chars','num_of_words','num_of_sent']].describe()

# describe these cols for ham sms/email
df[df['output']==0][['num_of_chars','num_of_words','num_of_sent']].describe()

# describe these cols for spam sms/email
df[df['output']==1][['num_of_chars','num_of_words','num_of_sent']].describe()

import seaborn as sns
sns.histplot(df[df['output']==0]['num_of_chars'],color='blue')
sns.histplot(df[df['output']==1]['num_of_chars'],color='red')
plt.title('Distrubtion of num of chars ')
plt.show()

import seaborn as sns
sns.histplot(df[df['output']==0]['num_of_words'],color='blue')
sns.histplot(df[df['output']==1]['num_of_words'],color='red')
plt.title('Distrubtion of num of words ')
plt.show()

import seaborn as sns
sns.histplot(df[df['output']==0]['num_of_sent'],color='blue')
sns.histplot(df[df['output']==1]['num_of_sent'],color='red')
plt.title('Distrubtion of num of sents ')
plt.show()

df2=df[['output','num_of_chars','num_of_words','num_of_sent']]
df2

sns.heatmap(df2.corr(),annot=True)
plt.show()

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('dancing')

from nltk.corpus import stopwords
stopwords.words('english')

def process_txt(txt):
    txt=txt.lower() # cvt all sms to lower casechase letters
    txt=nltk.word_tokenize(txt)
    
    y=[]
    for i in txt: #include nums
        if i.isalnum()==True:
            y.append(i)
    txt=y[:]
    y.clear()
    
    for i in txt:
        if i not in stopwords.words('english')and i not in string.punctuation: # ignore stopwords & punctuation
            y.append(i)
    txt=y[:]
    y.clear()
    for i in txt:
        y.append(ps.stem(i)) #get stemming
    
    
    return " ".join(y)

txt="hello 20 %"
process_txt(txt)

print(df['sms/email'][2000])
print('_'*100)
print(process_txt(df['sms/email'][2000]))

df['text_transform']=df['sms/email'].apply(process_txt)
df

from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,min_word_length=4,background_color='black')

# get the most frequently ham words
ham_words=wc.generate(df[df['output']==0]['text_transform'].str.cat(sep=" "))
plt.imshow(ham_words)
plt.show()

# get the most frequently spam words
spam_words=wc.generate(df[df['output']==1]['text_transform'].str.cat(sep=" "))
plt.imshow(spam_words)
plt.show()

spam_corpus=[]
for msg in df[df['output']==1]['text_transform'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        
        
# get the 50 most frequent words in spam emails/SMS
from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(50))

ham_corpus=[]
for msg in df[df['output']==0]['text_transform'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
        
        
# get the 50 most frequent words in ham emails/SMS        
from collections import Counter
pd.DataFrame(Counter(ham_corpus).most_common(50))

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

#feature selection
X = tfidf.fit_transform(df['text_transform']).toarray()
y=df['output'].values

from sklearn.model_selection import train_test_split 

#splitting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=1,stratify=y,shuffle=True)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.naive_bayes import GaussianNB ,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report
# MultinomialNB Model
gnb=GaussianNB()

gnb.fit(X_train,y_train)
y_pred_gnb=gnb.predict(X_test)

print(accuracy_score(y_test,y_pred_gnb))
print(confusion_matrix(y_test,y_pred_gnb))
print(classification_report(y_test,y_pred_gnb))

cm = confusion_matrix(y_test, y_pred_gnb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

bnb=BernoulliNB()


bnb.fit(X_train,y_train)
y_pred_bnb=bnb.predict(X_test)

print(accuracy_score(y_test,y_pred_bnb))
print(confusion_matrix(y_test,y_pred_bnb))
print(classification_report(y_test,y_pred_bnb))

cm = confusion_matrix(y_test, y_pred_bnb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#saving the model
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))