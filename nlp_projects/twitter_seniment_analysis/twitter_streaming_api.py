from tweepy import OAuthHandler
from SentimentAnalysis import sentiment_mod as m
# consumer key, consumer secret, access token, access secret.

ckey = "9mUcji3aFQEUMSxt681I5bH1O"
csecret = "Kl3FoaMZJ4fio4R2QT7X80oHBBR0GiQHyX1EveK6f24TjSbgPP"
atoken = "2829327546-hIomrAIj7aFYhm1D6PTY8OVZonWelUMwua5UPSZ"
asecret = "XPS3oUNshx8xqgZuEO2BzweJMpBNcUDumKhLyH7hEgmOO"



auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

import tweepy
api = tweepy.API(auth)

tweets = api.search(q='Modi',  count = 200)
for tweet in tweets:
    if tweet.lang=='en':
        print("###########################################################")
        print(tweet.text)
        print(m.sentiment(tweet.text))