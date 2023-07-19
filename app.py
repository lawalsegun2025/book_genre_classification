import pickle 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re 
import nltk 
from flask import Flask, rquest, render_template

# cleaning the text data (removing all unnecessary characters)

def clean_text(text):

    # removing the "\" sign
    text = re.sub("'\''","",text)

    # remove special characters an leaving only text(both lower and upper)
    text = re.sub("[^a-zA-Z]", " ", text)

    # Remove white spaces
    text = " ".join(text.split())

    # Conevrt text to lowercase
    text = text.lower()

    return text

# remove stopwords 
def remove_stop_words(text):
    
    stop_words = set(stopwords.words("english"))

    removed_stop_word = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(removed_stop_word)

# lemmatizing the text
def lemmatizing(text): 

    lemma = WordNetLemmatizer()

    text = text.split()
    
    # lemmatize
    text = [lemma.lemmatize(word) for word in text]
    
    return " ".join(text)

