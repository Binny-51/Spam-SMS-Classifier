import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Initialize the stemmer and tokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)   # ✅ replaced word_tokenize with RegexpTokenizer
    
    y = []
    # Keep only alphanumeric tokens (RegexpTokenizer already does this, but kept for safety)
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load trained vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
