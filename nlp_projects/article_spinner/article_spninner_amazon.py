import nltk
import random
import numpy as np

from bs4 import BeautifulSoup
from nltk.compat import xrange

positive_reviews = BeautifulSoup(open("positive.review").read())
positive_reviews = positive_reviews.findAll('review_text')

# extract the trigrams
# Key -> first and last word
# value -> possible middle words

trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in xrange(len(tokens)-2):
        k = (tokens[i],tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# turn each array of middle words into a probability vector
trigram_probabilities = {}
for k, words in trigrams.items():
    # create a dictionary of words -> count
    if len(set(words)) > 1:
        # only do this when there are different possibilities for a middle word
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigram_probabilities[k] = d

# means if possible middle words were ["cat", "cat", "dog"]
# then we would now have a dictionary with {"cat":.67 , "dog":0.33}

# way to randomly sample the dictionary


def random_sample(d):
    # choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    rand_review = random.choice(positive_reviews)
    s = rand_review.text.lower()
    print("Original:",s)
    word_tokens = nltk.tokenize.word_tokenize(s)
    for index in xrange(len(word_tokens)-2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (word_tokens[index], word_tokens[index + 2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                word_tokens[index + 1] = w

    print("Spun:"," ".join(word_tokens).replace(" .",".").
                                        replace(" '","'").
                                        replace(" ,",",").
                                        replace("$ ","$").replace(" !","!"))

if __name__ == '__main__':
    test_spinner()







