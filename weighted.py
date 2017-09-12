import os 
import numpy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

dataset_name = 'dataset.txt'
dataset_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
test_size = .3
show_count = 50

lines = [l.strip() for l in open(dataset_path)]
sentences = [l.split('\t')[0] for l in lines]
labels = [l.split('\t')[1] for l in lines]

cvectorizer = CountVectorizer(ngram_range=(1, 4), lowercase=True)
X = cvectorizer.fit_transform(sentences)
vocabularies = cvectorizer.vocabulary_.keys()
cvectorizer = CountVectorizer(vocabulary=vocabularies, ngram_range=(1, 3), lowercase=True)
X = cvectorizer.fit_transform(sentences)
print("The input [%s] file has %d observations with %d features" % (dataset_name, X.shape[0], X.shape[1]))
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=10)

clf = LinearSVC()
clf.fit(x_train, y_train)
w = clf.coef_[0]
wv = zip(w, vocabularies)
wv.sort()
print("Showing top %d positive words:" % show_count)
for line in wv[-show_count:]:
    print("score: %.2f, [%s]" % (line[0], line[1]))
print("\nShowing top %d negative words:" % show_count)
for line in wv[:show_count]:
    print("score: %.2f, [%s]" % (line[0], line[1]))

