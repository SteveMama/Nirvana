from flask import Flask, request, render_template
from splitter import splitter
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from typing import List
#from google.colab import files
import regex as re
import os, wave, math, collections
from os import path
from pydrive.drive import GoogleDrive
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import torch.quantization
import torch
from transformers import AutoConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords


#load tokenizer and model
tok = PegasusTokenizer.from_pretrained('C:/Users/Ramana/Desktop/pegasus/nirvana-test/SMA/models')
config = AutoConfig.from_pretrained('C:/Users/Ramana/Desktop/pegasus/nirvana-test/SMA/models')
dummy_model = PegasusForConditionalGeneration(config)


quantized_model = torch.quantization.quantize_dynamic(
    dummy_model, {torch.nn.Linear}, dtype=torch.qint8
)

quantized_state_dict = torch.load('C:/Users/Ramana/Desktop/pegasus/nirvana-test/SMA/models/pegasus-quantized.h5')
quantized_model.load_state_dict(quantized_state_dict)


app = Flask(__name__)



@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    #convert to lowercase
    text1 = request.form['text1']

    new_transcript = text1.replace("\n", " ")

    print(new_transcript)

    batch = tok.prepare_seq2seq_batch(src_texts=[new_transcript], truncation=True, padding='longest', return_tensors='pt')
    gen = quantized_model.generate(**batch)
    summary: List[str] = tok.batch_decode(gen,skip_special_tokens=True)
    new_summary = re.sub(r'[^\x00-\x7f]+', ' ', str(summary))

    print(new_summary)
    text_final = ''.join(c for c in text1 if not c.isdigit())
    


    return render_template('form.html',text1=new_summary)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
