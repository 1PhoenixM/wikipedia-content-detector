# Melissa Iori
# Natural Language Processing

# Loads previously trained model from trainer.py.
# Django gets the user's input.
# Model returns a classified version.

import os
import requests

import nltk
import pickle
import re
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et

from django.shortcuts import render
from django.http import HttpResponse

from .models import Greeting

msg= 'Promotional content is a big problem on Wikipedia. Enter the URL of a Wikipedia article below to check for promotional content.'
button= '<button type="submit">Test</button>'
fullform= '<h1>Wikipedia Promotional Content Detector</h1><br />'+msg+'<br /><form method="POST" action="/classify"><input name="article" type="text">'+button+'</form>'

template_message = 'This article contains content that is written like an advertisement. Please help improve it by removing promotional content and inappropriate external links, and by adding encyclopedic content written from a neutral point of view.'
template_message_part = '(Learn how and when to remove this template message)'
bigram_mincount = 2

# Create your views here.
def index(request):
    r = requests.get('http://httpbin.org/status/418')
    print(r.text)
    times= int(os.environ.get('TIMES', 3))
    #return HttpResponse(fullform)
    return render(request, 'index.html')

def classify(request):
    if request.method == 'POST':
        article = request.POST.get('article', None)
        file = open('bayes_2gram.pickle', 'rb')
        classifier = pickle.load(file)
        file.close()
        label = classifyArticle(classifier,article);
        res = article + " - " + label
        if label.strip() == 'Good':
            res = res + ' - No promotional content detected.'
        else:
            res = res + ' - Promotional content detected.'
        return render(request, 'classify.html', {'result': res})
    else:
        return render(request, 'index.html')

def db(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, 'db.html', {'greetings': greetings})
# Function to clean punctuation out of an article text
def cleanArticle(article):
    article = article.replace(template_message, '');
    article = article.replace(template_message_part, '');
    article = re.sub(r"(\[[0-9]+\])", "", article)
    article = article.replace("\n", " ")
    article = article.replace("\\n", " ")
    article = article.replace(",", "")
    article = article.replace(".", "")
    article = article.replace(";", "")
    return article

# Constructs a bag of words out of bigrams
def findBigrams(wordlist):
    featureset = {}
    for w in range(0, len(wordlist)-1):
        word_one = wordlist[w]
        word_two = wordlist[w+1]
        bigram = word_one + ' ' + word_two
        if bigram in featureset:
            featureset[bigram] += 1
        else:
            featureset[bigram] = 1
    for bigram in list(featureset.keys()):
        if featureset[bigram] < bigram_mincount:
            del featureset[bigram]
    return featureset

def classifyArticle(classifier,url):
    full_res = requests.get(url)
    article_html = et.fromstring(full_res.content)
    fullarticle = str(full_res.content).encode('utf-8').strip()
    parsed_html = BeautifulSoup(fullarticle, 'lxml')
    fulltext = parsed_html.body.find('div', attrs={'class':'mw-content-ltr'}).text
    fulltext = cleanArticle(fulltext)
    wordlist = fulltext.split(" ")
    featureset = findBigrams(wordlist)
    featureset["article_length"] = len(wordlist)
    label = classifier.classify(featureset)
    return label