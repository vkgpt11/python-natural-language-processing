import pandas as pd
import sklearn
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import itertools


df = pd.read_csv("fake_or_real_news.csv")

# print("Shape of data:", df.shape)
# print("First Line \n", df.iloc[3])

df = df.set_index("Unnamed: 0")
# print(df.head())

y = df.label

df.drop("label",axis=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33,
                                                                        random_state=53)

# Building Vectorizer Classifiers
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# print(tfidf_vectorizer.get_feature_names()[-10:])
# print(count_vectorizer.get_feature_names()[:10])
print(len(count_train.A),":",len(count_vectorizer.get_feature_names()))
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
#
difference = set(count_df.columns) - set(tfidf_df.columns)
difference
print(count_df.equals(tfidf_df))
count_df.head()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

from  sklearn.naive_bayes import MultinomialNB
import numpy as np
clf = MultinomialNB()

# clf.fit(tfidf_train, y_train)
# pred = clf.predict(tfidf_test)
# score = sklearn.metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# cm = sklearn.metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
# plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# clf.fit(count_train, y_train)
# pred = clf.predict(count_test)
# score = sklearn.metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# cm = sklearn.metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
# plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

linear_clf = PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = sklearn.metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = sklearn.metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# We can test if tuning the alpha value for a MultinomialNB creates comparable results.

# last_score = 0
# for alpha in np.arange(0,1,.1):
#     nb_classifier = MultinomialNB(alpha=alpha)
#     nb_classifier.fit(tfidf_train, y_train)
#     pred = nb_classifier.predict(tfidf_test)
#     score = sklearn.metrics.accuracy_score(y_test, pred)
#     if score > last_score:
#         clf = nb_classifier
#     print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))

# it might be interesting to perform parameter tuning across all of the classifiers,
#  or take a look at some other scikit-learn Bayesian classifiers.

def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    See: https://stackoverflow.com/a/26980472
    
    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not 
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


# most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)

feature_names = tfidf_vectorizer.get_feature_names()
sorted(zip(linear_clf.coef_[0], feature_names), reverse=True)[:20]
