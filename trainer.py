# trainer.py
# Melissa Iori
# Natural Language Processing

# Gets training set and trains promotional content model for Wikipedia articles.
# Using the module 'pickle', the model is saved for quick use by a Web front-end.

import sys
import nltk
import pickle
import requests
from random import shuffle
import xml.etree.ElementTree as et
import html2text
from bs4 import BeautifulSoup
import re

cat_api = 'https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&format=json&cmtitle=Category:';
full_api = 'https://en.wikipedia.org/?curid=';
good_article_category = 'Featured articles' 
bad_article_category = 'Articles with a promotional tone from November 2017'
template_message = 'This article contains content that is written like an advertisement. Please help improve it by removing promotional content and inappropriate external links, and by adding encyclopedic content written from a neutral point of view.'
template_message_part = '(Learn how and when to remove this template message)'

# Capture arguments
# Example to use Max Entropy classifier, train for 15 iterations on 50 articles with bi-grams:
# python trainer.py 2 50 15 maxent

# ngram_size should be 1-4
# training_set_size and iterations_to_use can be variable
# classifier_to_use can be "maxent", "bayes", or "decisiontree"
if len(sys.argv) < 4:
	print('Error - Needs 4 arguments: the n-gram size, number of articles to train on, number of training iterations, and the classifier to use - maxent, bayes or decisiontreemq.')
	sys.exit()

ngram_size = sys.argv[1]
training_set_size = sys.argv[2]
iterations_to_use = sys.argv[3]
classifier_to_use = sys.argv[4]

# The accuracy will be reported in this string
accuracy = "";

# To decrease the number of features, I placed limits on the acceptable n-gram count.
# N-grams that appear too few or too many times are not included in the featureset.
word_mincount = 3
word_maxcount = 30
bigram_mincount = 2

# Function to get an article
def getArticleList(type, number):
	req = cat_api + bad_article_category
	if type == 'good':
		req = cat_api + good_article_category
	res = requests.get(req)
	members = res.json()['query']['categorymembers']
	articleList = []
	for x in range(0, number):
		datapoint = {}

		pid = members[x]['pageid']
		full_req = full_api + str(pid)
		full_res = requests.get(full_req)
		
		#print(full_res.content)
		article_html = et.fromstring(full_res.content)
		fullarticle = str(full_res.content).encode('utf-8').strip()
		parsed_html = BeautifulSoup(fullarticle, 'lxml')
		# Unicode chars are ok as long as you don't print them
		#print(parsed_html.body.find('div', attrs={'class':'mw-content-ltr'}).text)
		#print(html2text.html2text(str(fullarticle)))
		#articleList.append(pid)
		datapoint["text"] = parsed_html.body.find('div', attrs={'class':'mw-content-ltr'}).text
		# The training examples have already been marked with a message from Wikipedia
		# We need to remove this message so it is not used as a feature!
		datapoint["text"] = cleanArticle(datapoint["text"])
		if type == 'good':
			datapoint["label"] = 'Good'
		else:
			datapoint["label"] = 'Bad' 
		articleList.append(datapoint)
	return articleList	

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

# Set up training and test sets
training_data_articles = [] 
test_data_articles = []

good_articles = getArticleList('good', int(training_set_size))
bad_articles = getArticleList('bad', int(training_set_size))

tsize = int(training_set_size)
mid = tsize//2

for a in range(0, mid):
	training_data_articles.append(good_articles[a])

for b in range(mid, tsize):
	training_data_articles.append(bad_articles[b])

for c in range(mid, tsize):
	test_data_articles.append(good_articles[c])

for d in range(0, mid):
	test_data_articles.append(bad_articles[d])

shuffle(training_data_articles)
shuffle(test_data_articles)

training_data = []
test_data = []
accuracy_test = []
counter = 0

# Wrangle training articles
for article in training_data_articles:
	point = ["", ""];
	point[0] = article["text"]
	point[1] = article["label"]
	training_data.append(point)
	counter += 1

# Wrangle test articles
for article in test_data_articles:
	test_data.append(article["text"])
	point = ["", ""];
	point[0] = article["text"]
	point[1] = article["label"]
	accuracy_test.append(point)
	counter += 1

# Convert raw text into feature set
for data in training_data:
	fulltext = data[0]
	wordlist = fulltext.split(" ")
	featureset = findBigrams(wordlist)
	featureset["article_length"] = len(wordlist)
	data[0] = featureset

# Convert test data text into featuresets
true_test_data = []
for x in range(0, len(test_data)):
	fulltext = test_data[x]
	wordlist = fulltext.split(" ")
	featureset = findBigrams(wordlist)
	featureset["article_length"] = len(wordlist)
	true_test_data.append(featureset)
	accuracy_test[x][0] = featureset	

# Train the classifier
if classifier_to_use == 'maxent':
	classifier = nltk.MaxentClassifier.train(training_data, max_iter=int(iterations_to_use))
elif classifier_to_use == 'bayes':
	classifier = nltk.NaiveBayesClassifier.train(training_data)
elif classifier_to_use == 'decisiontree':
	classifier = nltk.DecisionTreeClassifier.train(training_data)
else:
	print('Error - Unrecognized classifier, please use maxent, bayes or decisiontree as the 4th argument.')
	sys.exit()

# Classify the tests
'''
results = "";
for t in range(0, len(true_test_data)):
	results += classifier.classify(true_test_data[t]).strip() + "\n"
print(results)
'''

# Report the accuracy (via test set)
print(nltk.classify.accuracy(classifier, accuracy_test))

# Save model for later so we don't have to train it again
file = open(classifier_to_use + '_' + ngram_size + 'gram' + '.pickle', 'wb')
pickle.dump(classifier, file)
file.close()