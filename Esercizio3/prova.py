import pandas as pd
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline 
from nltk.lm import MLE
from nltk.util import ngrams
import random


# Sostituire con il proprio set di testi dei tweet da Trump Twitter Archive
df = pd.read_csv('Esercizio3\\trump_twitter_archive\\tweets.csv', sep=',')

tokenized_sent=[]
filtered_tokens=[]

for ind in df.index:
    
    sentences = df.loc[:, 'text'][ind]
    tokenized_sent.append(nltk.word_tokenize(str(sentences))) 
    sapo = [word.lower() for word in tokenized_sent[ind] if word.isalpha()]
    filtered_tokens = filtered_tokens +sapo



#corpus = "In this beautiful country. Great news today. Make America great again. ..."

# Preprocessa il corpus#
#tokens = corpus.split()

# Crea bi-grammi
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, filtered_tokens)

# Addestra un modello MLE su bi-grammi
mle = MLE(n)
mle.fit(train_data, padded_sents)

# Funzione per generare un tweet basato su bi-grammi con MLE
def generate_bi_gram_tweet_mle():
    tweet = []
    current_word = random.choice(filtered_tokens)
    tweet.append(current_word)
    
    while len(tweet) < 10:
        next_word = mle.generate (10, random_seed=4)
        tweet.append(next_word)
        current_word = next_word
    
    return ' '.join(str(tweet))

# Genera un tweet basato su bi-grammi con MLE
tweet = generate_bi_gram_tweet_mle()
print(tweet)