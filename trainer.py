# trainer.py
# Melissa Iori
# Natural Language Processing

# Gets training set and trains promotional content model for Wikipedia articles.
# Using the module 'pickle', the model is saved in a file for quick use by a Web front-end.

import sys
import nltk
import pickle
import requests
from random import shuffle
import io

import xml.etree.ElementTree as et
import html2text
from bs4 import BeautifulSoup
import re

# Gets all article IDs for a category.
cat_api = 'https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&format=json&cmtitle=Category:';

# Gets a page by ID
full_api = 'https://en.wikipedia.org/?curid=';

# Featured articles were used as "good" examples. These are the best examples available of what a Wikipedia article should be like.
# The latest marked promotional articles were used as "bad" examples. I used these because older examples might have been already fixed.
good_article_category = 'Featured articles' 
bad_article_category = 'Articles with a promotional tone from November 2017'

# The promotional articles already contain these template messages.
# I parse them out because it would bias the training set to expect these alerts to appear in any random article.
# Articles in question may not be marked with these template messages, so I remove them from the training input.
ad_message = 'This article contains content that is written like an advertisement.'
template_message = 'This article contains content that is written like an advertisement. Please help improve it by removing promotional content and inappropriate external links, and by adding encyclopedic content written from a neutral point of view.'
template_message_part = '(Learn how and when to remove this template message)'
date_message = '(November 2017)'
interesting_template_message = "This article may have been created or edited in return for undisclosed payments, a violation of Wikipedia's terms of use. It may require cleanup to comply with Wikipedia's content policies."
another_interesting_template_message = "This article reads like a press release or a news article or is entirely based on routine coverage. Please expand this article with properly sourced content to meet Wikipedia's quality standards, event notability guideline, or encyclopedic content policy."
bio_template_message = "This biographical article is written like a résumé. Please help improve it by revising it to be neutral and encyclopedic."
multiple_issues = "This article has multiple issues. Please help improve it or discuss these issues on the talk page. "
autobio_template_message = "This article is an autobiography or has been extensively edited by the subject or by someone connected to the subject."
yet_another_template_message = "A major contributor to this article appears to have a close connection with its subject."
catch_all = "This article"

# Capture arguments
# Example to use Max Entropy classifier, train on 50 articles with bi-grams:
# python trainer.py 2 50 maxent
# ngram_size should be 1-4
# training_set_size can be any number, upper limit being 21,558 (that's how many promotional articles are marked as of right now and may change)
# the more articles in the training set, the longer training will take
# classifier_to_use can be "maxent", "bayes", or "decisiontree"

if len(sys.argv) < 3:
	print('Error - Needs 3 arguments: the n-gram size, number of articles to train on, and the classifier to use - maxent, bayes or decisiontree.')
	sys.exit()

ngram_size = sys.argv[1]
training_set_size = sys.argv[2]
classifier_to_use = sys.argv[3]
sys.stdout.write("Using " + ngram_size + "-grams, getting " + training_set_size + " articles, and creating a " + classifier_to_use + " model...\n")
sys.stdout.flush()
 
# The accuracy will be reported in this string
accuracy = "";

# To decrease the number of features, I placed limits on the acceptable n-gram count.
# N-grams that appear too few or too many times are not included in the featureset.
word_mincount = 3
word_maxcount = 60
bigram_mincount = 2

# Hack: maxent crashes with overflow error if there are too many features...
if classifier_to_use == 'maxent':
	word_mincount = 5
	bigram_mincount = 3

# Function to get an article list. 
# The arguments:
# The type of articles to get - "good" or "bad"
# The total number of articles to get
# If the request is continuing, add the continue parameter to the next request
# The current list of articles being built up
def getArticleList(type, number, continueRequest, articleList):
	# Which article category to get, and whether or not it's continuing - the request is a URL
	req = cat_api + bad_article_category
	if type == 'good':
		req = cat_api + good_article_category
	if continueRequest != "":
		req += "&cmcontinue=" + continueRequest

	# Perform request to get JSON	
	res = requests.get(req)
	members = res.json()['query']['categorymembers']
	
	# For each article metadata
	for x in range(0, len(members)):
		datapoint = {}

		# The pageid uniquely references the article so we can get the full text
		pid = members[x]['pageid']
		full_req = full_api + str(pid)
		full_res = requests.get(full_req)
		
		
		#article_html = et.fromstring(full_res.content)
		
		# Convert the raw response into html
		fullarticle = str(full_res.content).encode('utf-8').strip()
		parsed_html = BeautifulSoup(fullarticle, 'lxml')
		
		# Unicode chars are ok as long as you don't print them
		# Find the content div - this contains the page's content text
		datapoint["text"] = parsed_html.body.find('div', attrs={'class':'mw-content-ltr'}).text

		# The training examples have already been marked with a message from Wikipedia
		# We need to remove this message so it is not used as a feature!
		datapoint["text"] = cleanArticle(datapoint["text"])
		
		# Set the label
		if type == 'good':
			datapoint["label"] = 'Good'
		else:
			datapoint["label"] = 'Bad' 
		
		# Add to list
		articleList.append(datapoint)
	
	# The Wikipedia API sends chunked responses back to save time
	# The cmcontinue parameter allows for another request to get more articles that are part of that category
	# getArticleList continues to call recursively until enough articles are found to train on
	if len(articleList) < number:
		getArticleList(type, number, res.json()['continue']['cmcontinue'], articleList)

	return articleList	

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

# Set up training and test sets
training_data_articles = [] 
test_data_articles = []

# Get a mix of good and bad article examples
sys.stdout.write("Getting some good article examples...\n")
sys.stdout.flush()
good_articles = getArticleList('good', int(training_set_size), "", [])
sys.stdout.write("Getting some promotional article examples...\n")
sys.stdout.flush()
bad_articles = getArticleList('bad', int(training_set_size), "", [])

# Here, the training set gets the first half of good articles and the second half of bad articles
# Likewise, the test set gets the first half of bad articles and the second half of good articles
tsize = int(training_set_size)
mid = tsize//2

sys.stdout.write("Building datasets...\n")
sys.stdout.flush()
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

# These are the actual data lists
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
sys.stdout.write("Processing data...\n")
sys.stdout.flush()
for data in training_data:
	fulltext = data[0]
	wordlist = fulltext.split(" ")
	if ngram_size == '4':
		featureset = findQuadgrams(wordlist)
	elif ngram_size == '3':
		featureset = findTrigrams(wordlist)
	elif ngram_size == '2':
		featureset = findBigrams(wordlist)
	else:
		featureset = findWords(wordlist)
	featureset["article_length"] = len(wordlist)
	data[0] = featureset

# Convert test data text into featuresets
true_test_data = []
for x in range(0, len(test_data)):
	fulltext = test_data[x]
	wordlist = fulltext.split(" ")
	if ngram_size == '4':
		featureset = findQuadgrams(wordlist)
	elif ngram_size == '3':
		featureset = findTrigrams(wordlist)
	elif ngram_size == '2':
		featureset = findBigrams(wordlist)
	else:
		featureset = findWords(wordlist)
	featureset["article_length"] = len(wordlist)
	true_test_data.append(featureset)
	accuracy_test[x][0] = featureset	

# Train the classifier
sys.stdout.write("Training on examples...\n")
sys.stdout.flush()
if classifier_to_use == 'maxent':
	classifier = nltk.MaxentClassifier.train(training_data, max_iter=15)
elif classifier_to_use == 'bayes':
	classifier = nltk.NaiveBayesClassifier.train(training_data)
elif classifier_to_use == 'decisiontree':
	classifier = nltk.DecisionTreeClassifier.train(training_data)
else:
	sys.stdout.write('Error - Unrecognized classifier, please use maxent, bayes or decisiontree as the 4th argument.\n')
	sys.stdout.flush()
	sys.exit()

# Report the accuracy (via test set)
sys.stdout.write("***Final accuracy is: ***\n")
sys.stdout.flush()
sys.stdout.write(str(nltk.classify.accuracy(classifier, accuracy_test)) + "\n")
sys.stdout.flush()
if classifier_to_use != 'decisiontree':
	sys.stdout.write("***10 best features: ***\n")
	sys.stdout.flush()
	classifier.show_most_informative_features(10)
	
# Save model for later so we don't have to train it again in the front end
sys.stdout.write("Saving model to " + classifier_to_use + '_' + ngram_size + 'gram' + training_set_size + '.pickle' + "...\n")
sys.stdout.flush()
file = open(classifier_to_use + '_' + ngram_size + 'gram' + training_set_size + '.pickle', 'wb')
pickle.dump(classifier, file)
file.close()
sys.stdout.write("Done!\n")
sys.stdout.flush()