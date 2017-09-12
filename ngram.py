# ngram, normalization, linear SVM

import os 
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


dataset_name = 'dataset.txt'
dataset_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
test_size = .3

lines = [l.strip() for l in open(dataset_path)]
sentences = [l.split('\t')[0] for l in lines]
labels = [l.split('\t')[1] for l in lines]

cvectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2))
X = cvectorizer.fit_transform(sentences)
normalize(X)
print("The input [%s] file has %d observations with %d features" % (dataset_name, X.shape[0], X.shape[1]))
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)

##  Guassian naive bayes classifier  ##
clf = GaussianNB()
clf.fit(x_train.toarray(), y_train)
pre_y = clf.predict(x_test.toarray())
print("Accuracy Score for test size of %.1f" % test_size)
print("Gaussian Naive Bayes: %f" % accuracy_score(y_pred=pre_y, y_true=y_test))

## Linear SVM ##
clf = GaussianNB()
clf.fit(x_train.toarray(), y_train)
pre_y = clf.predict(x_test)
print("Accuracy Score for test size of %.1f" % test_size)
print("Linear SVM: %f" % accuracy_score(y_pred=pre_y, y_true=y_test))