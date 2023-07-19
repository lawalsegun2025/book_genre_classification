import pickle 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re 
import nltk 
from flask import Flask, request, render_template

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

# stemming the text
def stemming(text):
    stemmer = PorterStemmer()

    text = text.split()
    
    # stem
    text = [stemmer.stem(word) for word in text]
    
    return " ".join(text)


# testing the model
def test(text, model, tfidf_vectorizer):
    
    text = clean_text(text)
    text = remove_stop_words(text)
    text = lemmatizing(text)
    text = stemming(text)
    
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    new_mapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    
    return new_mapper[predicted[0]]

# Loading the model and tfidf vectorizer
file = open("book_genre_model.pkl", "rb")
model = pickle.load(file)
file.close()

file_1 = open("tfidf_vector.pkl", "rb")
tfidf_vectorizer = pickle.load(file_1)
file_1.close()


app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():

    if request.method == "POST":

        my_dict = request.form
        text = my_dict["summary"]
        prediction = test(text, model, tfidf_vectorizer)

        return render_template('index.html', genre=prediction, text=str(text)[:100], show_result=True)
    return render_template("index.html")

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)