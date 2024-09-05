import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))


def preprocess(text):
  y = []
  text = text.lower()
  text = nltk.word_tokenize(text)
  for i in text:
    if i.isalnum():
      if i not in stopwords.words("english") and i not in string.punctuation:
        y.append(ps.stem(i))
  return " ".join(y)

st.title('Email/SMS Spam Classifier')
text = st.text_input(label='Enter the message')

if st.button('Predict'):
    #preprocess
    transformed_text = preprocess(text)
    #vectorize
    vectorText = tfidf.transform([transformed_text])
    #model use
    prediction = model.predict(vectorText)[0]
    #result
    if prediction == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")