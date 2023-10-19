import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline 
from nltk.lm import MLE
from nltk import word_tokenize

trump = pd.read_csv('Esercizio3\\trump_twitter_archive\\tweets.csv')
sentences = list(trump['text'].apply(word_tokenize))
tokens = [[token for token in sent if token.isalpha() ] for sent in sentences]


train, vocab = padded_everygram_pipeline(2, sentences)
tweet_like_a_Trump = MLE(2)
tweet_like_a_Trump.fit(train, vocab)

print(tweet_like_a_Trump.generate(10))



train, vocab = padded_everygram_pipeline(3, sentences)
tweet_like_a_Trump = MLE(3)
tweet_like_a_Trump.fit(train, vocab)

print(tweet_like_a_Trump.generate(9))







