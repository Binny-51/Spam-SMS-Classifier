import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer


nltk_resources = [
    'stopwords',
    'punkt',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'vader_lexicon'
]

for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize the stemmer and tokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)
    
    y = []
    # Keep only alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    for i in text:
        if i not in stop_words:
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
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
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
