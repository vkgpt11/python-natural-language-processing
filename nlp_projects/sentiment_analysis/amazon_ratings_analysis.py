# Data set
# http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
# tuning a word to its base form. eg dogs to dog
wordnet_lemmatizer = WordNetLemmatizer()

# don't include stopwords

stop = set(stopwords.words('english')) # set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open("amazon_electronic/positive.review").read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open("amazon_electronic/negative.review").read())
negative_reviews = negative_reviews.findAll('review_text')

# since there are more positive reviews then negative, for some classifiers this is not a problem,
# but lets be safe and do it anyway
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


# tokenize
def my_tokenizer(s):
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (token)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they are probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stop]  # remove stopwords
    return tokens


# Note: NLTK's tokenization function is much slower the string.split() so you may wants to explore other
# options in your code.

# Next, we need to collect the data necessary in order to turn all the text into a data matrix X.
# Includes: finding total size of vocabulary , determining the index of each word, and saving each review
# in tokenize form

word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

# Next, let's create a function to convert each set of tokens representing one review into a data vector


def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map)+1) # last element is for the label
    # print(x)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
        x = x/x.sum() # normalize it before setting label
        x[-1] = label
        return x


# Finally, We initialize our data matrix
N = len(positive_tokenized) + len(negative_tokenized)
# (N * D+1 matrix -keeping them together for now so we can shuffle more easily later
data = np.zeros((N,len(word_index_map)+1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens,1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens,0)
    data[i,:] = xy
    i += 1

np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]


# model = LogisticRegression()
# model.fit(Xtrain,Ytrain)
# print("Classification rate:",model.score(Xtest,Ytest))


# threshold = 0.5
# for word, index in word_index_map.items():
#     weight = model.coef_[0][index]
#     if weight > threshold :
#         print(word, " ", weight)


model = RandomForestClassifier()
model.fit(Xtrain,Ytrain)
print("Classification rate:",model.score(Xtest,Ytest))