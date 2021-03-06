from flask import Flask, request, render_template
from splitter import splitter
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from typing import List
#from google.colab import files
import regex as re
import os, wave, math, collections
from os import path
# from pydrive.drive import GoogleDrive
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
import shutil
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json
import re
import os
import pickle
from punctuator import Punctuator
import urllib.parse as urlparse


main_dir = "C:/Users/prash/Desktop/SMA/SMA/paras"

if os.path.exists(main_dir):
    shutil.rmtree(main_dir)  # delete output folder
os.makedirs(main_dir)


#load tokenizer and model
tok = PegasusTokenizer.from_pretrained('C:/Users/prash/Desktop/SMA/SMA/models')
config = AutoConfig.from_pretrained('C:/Users/prash/Desktop/SMA/SMA/models')
dummy_model = PegasusForConditionalGeneration(config)

#quantized model deployment
quantized_model = torch.quantization.quantize_dynamic(
    dummy_model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_state_dict = torch.load('C:/Users/prash/Desktop/SMA/SMA/models/pegasus-quantized.h5')
quantized_model.load_state_dict(quantized_state_dict)

app = Flask(__name__)

#creating the punctuator
p = Punctuator('C:/Users/prash/Desktop/SMA/SMA/punctuator/INTERSPEECH-T-BRNN.pcl')

def youtube_punctuation(id):
    video_id = id
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(transcript)

    with open('C:/Users/prash/Desktop/SMA/SMA/your_filename.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_formatted)

    def extraction(data):
        data = data
        f = open("C:/Users/prash/Desktop/SMA/SMA/transcript.txt", "a")
        for t in data:
            finall = t['text']
            f.write(finall)
        f.close()

    with open('C:/Users/prash/Desktop/SMA/SMA/your_filename.json') as json_file:
        data = json.load(json_file)

    extraction(data)

    with open('C:/Users/prash/Desktop/SMA/SMA/transcript.txt','r') as file:
        transcript = file.read()

    new_transcript = p.punctuate(transcript)

    return new_transcript



def preparing_transcript(text):

    transcript = text
    n = len(transcript.split())

    if n<7000:
        print("a")
        AR = open('C:/Users/prash/Desktop/SMA/SMA/test.txt', 'r').read()
        batch = tok.prepare_seq2seq_batch(src_texts=[AR], truncation=True, padding='longest', return_tensors='pt')
        gen = quantized_model.generate(**batch)
        summary: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
        print("0")
        sum = str(summary)

    elif 7000<n<14000:
        print("b")
        n_words = (len(transcript.split())//1)
        input = 'C:/Users/prash/Desktop/SMA/SMA/test.txt'
        output = 'C:/Users/prash/Desktop/SMA/SMA/paras'
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split0.txt', 'r').read()
        AR2 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split1.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')



        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)

        print("1")
        sum = str(summary1+summary2)

    elif 14000<n<21000:
        print("c")
        n_words = (len(transcript.split())//2)
        input = 'C:/Users/prash/Desktop/SMA/SMA/test.txt'
        output = 'C:/Users/prash/Desktop/SMA/SMA/paras'
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split0.txt', 'r').read()
        AR2 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split1.txt', 'r').read()
        AR3 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split2.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')


        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)

        print("2")
        sum = str(summary1+summary2+summary3)


    elif 21000<n<28000:
        print("d")
        n_words = (len(transcript.split())//3)
        input = 'C:/Users/prash/Desktop/SMA/SMA/test.txt'
        output = 'C:/Users/prash/Desktop/SMA/SMA/paras'
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split0.txt', 'r').read()
        AR2 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split1.txt', 'r').read()
        AR3 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split2.txt', 'r').read()
        AR4 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split3.txt', 'r').read()

        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)

        print("3")
        sum = str(summary1+summary2+summary3+summary4)

    elif 35000<n<42000:
        print("e")
        n_words = (len(transcript.split())//4)
        input = 'C:/Users/prash/Desktop/SMA/SMA/test.txt'
        output = 'C:/Users/prash/Desktop/SMA/SMA/paras'
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split0.txt', 'r').read()
        AR2 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split1.txt', 'r').read()
        AR3 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split2.txt', 'r').read()
        AR4 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split3.txt', 'r').read()
        AR5 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split4.txt', 'r').read()

        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')
        batch5 = tok.prepare_seq2seq_batch(src_texts=[AR5], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)
        gen5 = quantized_model.generate(**batch5)
        summary5: List[str] = tok.batch_decode(gen5, skip_special_tokens=True)

        print("4")
        sum = str(summary1+summary2+summary3+summary4+summary5)

    else:
        print("f")
        n_words = (len(transcript.split())//5)
        input = 'C:/Users/prash/Desktop/SMA/SMA/test.txt'
        output = 'C:/Users/prash/Desktop/SMA/SMA/paras'
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split0.txt', 'r').read()
        AR2 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split1.txt', 'r').read()
        AR3 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split2.txt', 'r').read()
        AR4 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split3.txt', 'r').read()
        AR5 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split4.txt', 'r').read()
        AR6 = open('C:/Users/prash/Desktop/SMA/SMA/paras/test_split5.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')
        batch5 = tok.prepare_seq2seq_batch(src_texts=[AR5], truncation=True, padding='longest', return_tensors='pt')
        batch6 = tok.prepare_seq2seq_batch(src_texts=[AR6], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)
        gen5 = quantized_model.generate(**batch5)
        summary5: List[str] = tok.batch_decode(gen5, skip_special_tokens=True)
        gen6 = quantized_model.generate(**batch6)
        summary6: List[str] = tok.batch_decode(gen6, skip_special_tokens=True)

        print("5")
        sum = str(summary1+summary2+summary3+summary4+summary5+summary6)

    return sum

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    text1 = request.form['text1']
    new_transcript = text1.replace("\n","")
    print(new_transcript)
    os.remove("C:/Users/prash/Desktop/SMA/SMA/test.txt")
    f = open("C:/Users/prash/Desktop/SMA/SMA/test.txt", "x")
    f.write(str(new_transcript))
    f.close()

    summary = preparing_transcript(new_transcript)
    new_summary = re.sub(r'[^\x00-\x7f]+',' ', str(summary))


    #youtube url-id extraction

    url = "take the youtube link from the flask application"
    url_data = urlparse.urlparse(url)
    query = urlparse.parse_qs(url_data.query)
    video = query["v"][0]
    utube_output = youtube_punctuation(video)
    utube_summary = preparing_transcript(utube_output)
    new_summary = re.sub(r'[^\x00-\x7f]+',' ', str(utube_summary))

    return render_template('form.html', text1=new_summary)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
