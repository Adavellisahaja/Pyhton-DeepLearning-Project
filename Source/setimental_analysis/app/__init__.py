from flask import Flask, redirect, url_for, request
from flask import render_template
from keras.models import load_model
import pickle,re,json
from keras.preprocessing.text import Tokenizer
import pandas as pd,numpy as np

def data_cleansing(string):
    letters_only = re.sub("[^a-zA-Z]", " ", string) 
    words = letters_only.lower().split()
    output =  re.sub(r"\b[a-zA-Z]\b", "", " ".join(words))                          
    return (output)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#now directly use tokenizer.texts_to_matrix(['your input string'])

model = load_model('senti_model.tf')
labels = ["Negative", "Positive"]
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query")
def query():
    text =  (request.args.get('text'))
    output = tokenizer.texts_to_matrix([text])
    output = model.predict_classes(output)
    print (output)
    output = labels[output[-1]]
    
    if text == '': text = 'Please enter some text'; output = ''
    print (text)
    return """
    <form action = "http://localhost:5000/query" method = "get">
         <p>Enter Text:</p>
         <p><input style="font-size: 11pt; height: 40px; width:280px;" type = "text" name = "text" /></p>
         <p><input type = "submit" value = "submit" /></p>
         <p>Input sentence:      <span>{}</span></p>
         <p>Output:             <span>{}</span></p>
      </form>
      """.format(text,output)

if __name__ == '__main__':
   app.run(debug = False)
