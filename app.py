from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
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
run_with_ngrok(app)




@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    #convert to lowercase
    text1 = request.form['text1']
    text2 = request.form['text2']
    text3 = request.form['text2']
    print(text1,text2,text3)
    new_transcript1 = text1.replace("\n", " ")
    new_transcript2 = text2.replace("\n"," ")
    new_transcript3 = text3.replace("\n"," ")


    print(new_transcript1)
    print(new_transcript2)
    print(new_transcript3)

    batch1 = tok.prepare_seq2seq_batch(src_texts=[new_transcript1], truncation=True, padding='longest', return_tensors='pt')
    batch2 = tok.prepare_seq2seq_batch(src_texts=[new_transcript2], truncation=True, padding='longest', return_tensors='pt')
    batch3 = tok.prepare_seq2seq_batch(src_texts=[new_transcript2], truncation=True, padding='longest', return_tensors='pt')
    gen1 = quantized_model.generate(**batch1)
    gen2 = quantized_model.generate(**batch2)
    gen3 = quantized_model.generate(**batch3)

    summary1: List[str] = tok.batch_decode(gen1,skip_special_tokens=True)
    summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
    summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)

    new_summary1 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary1))
    new_summary2 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary2))
    new_summary3 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary3))



    print(new_summary1)
    print(new_summary2)
    print(new_summary3)

    return render_template('form.html',text1=new_summary1, text2=new_summary2, text3=new_summary3)
if __name__ == "__main__":
    app.run()
