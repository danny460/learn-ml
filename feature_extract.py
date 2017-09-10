import os 
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


dataset_name = 'dataset.txt'
dataset_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
test_size = .3

lines = [l.strip() for l in open(dataset_path)]
sentences = [l.split('\t')[0] for l in lines]
labels = [l.split('\t')[1] for l in lines]

cvectorizer = CountVectorizer()
X = cvectorizer.fit_transform(lines)
print("The input [%s] file has %d observations with %d features" % (dataset_name, X.shape[0], X.shape[1]))
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)

##  Guassian naive bayes classifier  ##
gnb_clf = GaussianNB()
gnb_clf.fit(x_train.toarray(), y_train)
gnb_pred_y = gnb_clf.predict(x_test.toarray())
print("Accuracy Score for test size of %.1f" % test_size)
print("Gaussian Naive Bayes is: %f" % accuracy_score(y_pred=gnb_pred_y, y_true=y_test))

