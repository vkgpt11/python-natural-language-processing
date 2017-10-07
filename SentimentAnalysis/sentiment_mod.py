import pickle
from nltk.tokenize import word_tokenize

from SentimentAnalysis.VoteClassifier import VoteClassifier

documents_f = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("D:/Git/python-natural-language-processing/SentimentAnalysis/algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
    classifier,
    LinearSVC_classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
