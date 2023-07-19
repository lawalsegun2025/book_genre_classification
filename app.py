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

    # Conevrt text to lowercase
    text = text.lower()

    return text