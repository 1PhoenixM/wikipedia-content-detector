# backend.py
# Melissa Iori
# Natural Language Processing

# Loads previously trained model from trainer.py.
# Django gets the user's input.
# Model returns a classified version.

import nltk
import pickle

def parseArticle():
	return article

file = open('bayes_2gram.pickle', 'rb')
classifier = pickle.load(file)
file.close()

send = classifier.classify(parseArticle('NY'))