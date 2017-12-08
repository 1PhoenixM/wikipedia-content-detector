# views.py
# Melissa Iori
# Natural Language Processing

# Loads previously trained model file from trainer.py.
# Django gets the user's input.
# Model returns a decision - promotional article or not?

import os
import requests
import nltk
import numpy
import pickle
import re
import sys
import io

from bs4 import BeautifulSoup
import xml.etree.ElementTree as et

from django.shortcuts import render
from django.http import HttpResponse

# This is the model file to use - alter this variable to change to another model file
model_file = 'maxent_3gram100.pickle'

# Template messages need to be parsed out
template_message = 'This article contains content that is written like an advertisement. Please help improve it by removing promotional content and inappropriate external links, and by adding encyclopedic content written from a neutral point of view.'
template_message_part = '(Learn how and when to remove this template message)'
ad_message = 'This article contains content that is written like an advertisement.'
date_message = '(November 2017)'
interesting_template_message = "This article may have been created or edited in return for undisclosed payments, a violation of Wikipedia's terms of use. It may require cleanup to comply with Wikipedia's content policies."
another_interesting_template_message = "This article reads like a press release or a news article or is entirely based on routine coverage. Please expand this article with properly sourced content to meet Wikipedia's quality standards, event notability guideline, or encyclopedic content policy."
bio_template_message = "This biographical article is written like a résumé. Please help improve it by revising it to be neutral and encyclopedic."
multiple_issues = "This article has multiple issues. Please help improve it or discuss these issues on the talk page. "
autobio_template_message = "This article is an autobiography or has been extensively edited by the subject or by someone connected to the subject."
yet_another_template_message = "A major contributor to this article appears to have a close connection with its subject."
catch_all = "This article"

# Minimum appearance of bigrams (also trigrams)
bigram_mincount = 2

# Main view - index.html
def index(request):
    r = requests.get('http://httpbin.org/status/418')
    print(r.text)
    return render(request, 'index.html')

# POST /classify route performs the classification
# The template looks the same, only with extra output
def classify(request):
    if request.method == 'POST':
        # Get the article URL
        article = request.POST.get('article', None)
        
        # URL must contain http (or https) to be valid
        if "http" in article:

            # Load the previously trained model
            file = open(model_file, 'rb')
            classifier = pickle.load(file)
            file.close()

            # Classify new article
            finalres = classifyArticle(classifier,article);
            
            # Build output HTML
            res = ""
            if finalres["label"].strip() == 'Good':
                res = article + " <br /> <span style='color:green'>" + finalres["label"] + " - No promotional content detected.</span>"
            else:
                res = article + " <br /> <span style='color:red'>" + finalres["label"] + " - Promotional content detected.</span>"
            marked_article = "<div id='thetext'>" + finalres["article"] + "</div>"
            ngrams = "<h3>Most informative n-grams:</h3> <br />"
            best_features = ""

            # Bayesian classifiers return most informative features
            if 'bayes' in model_file:
                best_features = classifier.most_informative_features(20)
                for ngram in best_features:
                    marked_article = marked_article.replace(ngram[0], '<mark>' + ngram[0] + '</mark>')
                    ngrams += str(ngram[0]) + " = " + str(ngram[1]) + "<br />"
            
            # Maxent classifiers only PRINT the informative features, not return them
            # Have to redirect stdout in order to capture the information
            else:
                oldout = sys.stdout
                stream = io.StringIO()
                sys.stdout = stream
                classifier.show_most_informative_features(20)
                sys.stdout = oldout
                best_features = str(stream.getvalue())
                bestfeatures = best_features.split('\n')
                '''
                for ngram in bestfeatures:
                    parts = ngram.split(" ")
                    word_one = parts[1]
                    word_two = parts[2]
                    word_three = parts[3].split("==")[0]
                    real_ngram = word_one + ' ' + word_two + ' ' + word_three
                    marked_article = marked_article.replace(real_ngram, '<mark>' + real_ngram + '</mark>')
                '''
                # Nice formatting
                best_features = best_features.replace("'Good'", "'Good'<br />")
                ngrams += best_features
            # Render the output page
            return render(request, 'classify.html', {'result': res, 'ngrams': ngrams, 'articletext': '<h3>Article text:</h3> <br />' + marked_article})
        else:
            # If bad URL, render error page.
            return render(request, 'error.html', {})
    else:
        # If not POST, just render the main page
        return render(request, 'index.html')

# Function to clean punctuation and Wikipedia template messages and references out of an article text
def cleanArticle(article):
    article = article.replace(template_message, '');
    article = article.replace(ad_message, '');
    article = article.replace(template_message_part, '');
    article = article.replace(date_message, '');
    article = article.replace(interesting_template_message, '');
    article = article.replace(another_interesting_template_message, '');
    article = article.replace(bio_template_message, '');
    article = article.replace(multiple_issues, '');
    article = article.replace(autobio_template_message, '');
    article = article.replace(yet_another_template_message, '');
    article = article.replace(catch_all, '');

    # This is a reference regex. Wikipedia articles contain references in the form of: [1] which need to be removed
    article = re.sub(r"(\[[0-9]+\])", "", article) 
    
    article = article.replace("\n", " ")
    article = article.replace("\\n", " ")
    article = article.replace(",", "")
    article = article.replace(".", "")
    article = article.replace(";", "")
    return article

# Constructs a bag of words out of quadgrams
def findQuadgrams(wordlist):
    featureset = {}
    for w in range(0, len(wordlist)-3):
        word_one = wordlist[w]
        word_two = wordlist[w+1]
        word_three = wordlist[w+2]
        word_four = wordlist[w+3]
        quadgram = word_one + ' ' + word_two + ' ' + word_three + ' ' + word_four
        if quadgram in featureset:
            featureset[quadgram] += 1
        else:
            featureset[quadgram] = 1
    return featureset

# Constructs a bag of words out of trigrams
def findTrigrams(wordlist):
    featureset = {}
    for w in range(0, len(wordlist)-2):
        word_one = wordlist[w]
        word_two = wordlist[w+1]
        word_three = wordlist[w+2]
        trigram = word_one + ' ' + word_two + ' ' + word_three
        if trigram in featureset:
            featureset[trigram] += 1
        else:
            featureset[trigram] = 1
    for trigram in list(featureset.keys()):
        if featureset[trigram] < bigram_mincount:
            del featureset[trigram]
    return featureset

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

# Constructs a bag of words out of unigrams (single words)
def findWords(wordlist):
    featureset = {}
    for w in range(0, len(wordlist)):
        word = wordlist[w]
        if word in featureset:
            featureset[word] += 1
        else:
            featureset[word] = 1
    for word in list(featureset.keys()):
        if featureset[word] < word_mincount or featureset[word] > word_maxcount:
            del featureset[word]
    return featureset

# Parse, clean, process and classify a Wikipedia article from URL
def classifyArticle(classifier,url):
    # Extract and parse the given article
    full_res = requests.get(url)
    article_html = et.fromstring(full_res.content)
    fullarticle = str(full_res.content).encode('utf-8').strip()
    parsed_html = BeautifulSoup(fullarticle, 'lxml')
    fulltext = parsed_html.body.find('div', attrs={'class':'mw-content-ltr'}).text
    fulltext = cleanArticle(fulltext)
    wordlist = fulltext.split(" ")

    # Construct the appropriate featureset based on which model we're using
    featureset = {}
    if '1gram' in model_file:
        featureset = findWords(wordlist)
    elif '2gram' in model_file:
        featureset = findBigrams(wordlist)
    elif '3gram' in model_file:
        featureset = findTrigrams(wordlist)
    else:
        featureset = findQuadgrams(wordlist)
    featureset["article_length"] = len(wordlist)
    
    # Return the decision label and the text
    package = {}
    package["label"] = classifier.classify(featureset)
    package["article"] = fulltext
    return package