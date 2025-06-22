import pandas as pd 
import re 
import string 
import nltk 
from nltk.corpus import stopwords 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

file_pathv= "../data/raw/d3984639-b285-4eea-8665-06e109bca630.csv"
df = pd.read_csv(file_pathv)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ''.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

le= LabelEncoder()
df['label_encoded'] = le.fit_transform(df['target'])

X= df['cleaned_text']
y=df['label_encoded']

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_vect =tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)
print(classification_report(y_test,y_pred,target_names=[str(cls)for cls in le.classes_]))

joblib.dump(model,'../models/text_classifier.pkl')
joblib.dump(tfidf, '../models/vectorizer.pkl')
joblib.dump(le, '../models/label_encoder.pkl')

print("✅ All files saved successfully in '../models/'")
    
