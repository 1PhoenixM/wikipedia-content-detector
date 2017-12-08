# Wikipedia Promotional Content Detector

This is my COMP 150 NLP "Wikipedia Promotional Content Detector" project.

Read Pitch.pdf for more information.

Most extensive code is found in trainer.py and web/views.py.

[Live demo on Heroku](https://wikipedia-content-detector.herokuapp.com/)

Load time on this demo may be a little slow because it's on Heroku free tier.

Some examples that work not from training or testing sets:

Good:

https://en.wikipedia.org/wiki/Moon

https://en.wikipedia.org/wiki/Earth

Bad:

https://en.wikipedia.org/wiki/Varonis_Systems

https://en.wikipedia.org/wiki/The_Village_(studio)

Front-end code reference:

[Getting Started with Python on Heroku](https://github.com/heroku/python-getting-started)

## Running Back-End Locally

```sh
$ pip install requests
$ pip install html2text
$ pip install bs4
$ pip install numpy
$ pip install nltk
$ pip install lxml
$ python trainer.py <ngram-size> <training-set-size> <maxent | bayes | decisiontree> 
```

The ngram-size should be 1-4.

Training set size can be any number lower than 20,000, but this will take a very long time to obtain and parse. Recommended size is 20 - 200.

The final parameter is the classifier you want to use, one of three choices. The options are above.

This trainer will create a .pickle binary file containing the trained model.

The trained model filename must be manually specified in the front-end (directions to follow).

## Running Front-End Locally

```sh
$ pipenv install
$ python manage.py collectstatic
$ heroku local
```

Runs on localhost:5000.

To change the model used, place the chosen model file generated by trainer.py into the web/ directory.

In views.py, change the model_file variable to be equal to the filename.

## Deploying Front-end to Heroku

```sh
$ heroku create
$ git push heroku master
```

## Accuracy of Classifiers

The italicized accuracy is the model currently being used in the Web app. I chose to use a Maxent classifier that had been trained on 100 articles. (Accuracy 77%)The Bayes quadgram models showed good performance, but their file sizes are so large that it takes too long for the file to be loaded up in a Web app.

The bold accuracies indicate the model that had the best accuracy for that algorithm (where bayes is Naive Bayes, decisiontree is Decision Tree, and maxent is Maximum Entropy).

For Number of Training Articles, half are good, "featured article" examples, and the other half are bad, "promotional content" articles. The test set is the same size and same distribution, but with an entirely new set of articles that the classifier has not yet seen.

Maxent trained for 15 iterations every time. Some sessions did not converge.
Quadgram was not computed for maxent because the featureset contained many quadgrams, or four-grams, most of which only appeared once. The high amount of features caused the maxent trainer to have an overflow error. With other size n-grams, I restricted the amount of features by having a minimum and/or maximum frequency of n-gram appearance, but since most quadgrams only appear once or not at all, it's unclear how best to restrict that featureset and all quadgrams are equally considered.

| Classifier Type | N-gram Type | Number of Training Articles | Accuracy |
| --------------- | ----------- | --------------------------- | -------- |
| bayes 		  | Unigram (1) | 20						  |	70%		 |
| bayes 		  | Unigram (1) | 50						  |	68%		 |
| bayes 		  | Unigram (1) | 100						  |	70%		 |
| bayes			  | Bigram (2) | 20						  	  |	65%		 |
| bayes			  | Bigram (2) | 50						  	  |	60%		 |
| bayes			  | Bigram (2) | 100						  |	62%		 |
| bayes			  | Trigram (3) | 20						  |	70%		 |
| bayes			  | Trigram (3) | 50						  |	62%		 |
| bayes			  | Trigram (3) | 100						  |	67%		 |
| bayes	 		  | Quadgram (4) | 20						  |	**95%**  |
| bayes			  | Quadgram (4) | 50						  |	94%		 |
| bayes 		  | Quadgram (4) | 100						  |	92%	 |
| decisiontree 		  | Unigram (1) | 20					  |	**80%**	 |
| decisiontree 		  | Unigram (1) | 50					  |	50%		 |
| decisiontree 		  | Unigram (1) | 100					  |	50%		 |
| decisiontree 		  | Bigram (2) | 20						  |	70%		 |
| decisiontree 		  | Bigram (2) | 50						  |	78%		 |
| decisiontree 		  | Bigram (2) | 100				      |	50%		 |
| decisiontree 		  | Trigram (3) | 20					  |	50%		 |
| decisiontree 		  | Trigram (3) | 50					  |	50%		 |
| decisiontree 		  | Trigram (3) | 100				      |	50%		 |
| decisiontree 		  | Quadgram (4) | 20				      |	50%		 |
| decisiontree 		  | Quadgram (4) | 50					  | 50%	     |
| decisiontree 		  | Quadgram (4) | 100				      |	50%		 |
| maxent		  | Unigram (1) | 20						  |	50%		 |
| maxent		  | Unigram (1) | 50						  |	60%		 |
| maxent		  | Unigram (1) | 100						  |	65%		 |
| maxent		  | Bigram (2) | 20						  	  |	50%		 |
| maxent		  | Bigram (2) | 50						  	  |	62%		 |
| maxent		  | Bigram (2) | 100						  |	56%		 |
| maxent		  | Trigram (3) | 20						  |	65%		 |
| maxent		  | Trigram (3) | 50						  |	56%		 |
| maxent		  | Trigram (3) | 100						  |	***77%***	 |
