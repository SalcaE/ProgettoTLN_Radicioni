import random
import nltk
import pandas as pd
import string

# Assicurati di aver installato e scaricato i dati di NLTK


from nltk.util import bigrams
from nltk.probability import FreqDist, ConditionalFreqDist

df = pd.read_csv('Esercizio3\\trump_twitter_archive\\tweets.csv', sep=',')

tokenized_sent=[]
filtered_tokens=[]
alphabet = list(string.ascii_lowercase)
alphabet.pop(0)
for ind in df.index:
    
    sentences = df.loc[:, 'text'][ind]
    tokenized_sent.append(nltk.word_tokenize(str(sentences))) 
    sapo = [word.lower() for word in tokenized_sent[ind] if word.isalpha() and word not in alphabet]
    filtered_tokens = filtered_tokens +sapo




# Estrai i bigrammi
bi_grams = list(bigrams(filtered_tokens))

# Calcola le distribuzioni di frequenza
fdist = FreqDist(filtered_tokens)
cfdist = ConditionalFreqDist(bi_grams)

# Funzione per generare un tweet basato su bi-grammi con MLE
def generate_bi_gram_tweet_mle_varied():
    tweet = []
    current_word = random.choice(filtered_tokens)
    tweet.append(current_word)
    
    while len(tweet) < 7:
        next_options = list(cfdist[current_word].keys())
        next_probabilities = [cfdist[current_word].freq(word) for word in next_options]
        next_word = random.choices(next_options, next_probabilities)[0]
        tweet.append(next_word)
        current_word = next_word
    
    return ' '.join(tweet)

# Genera un tweet basato su bi-grammi con MLE e variazione
tweet = generate_bi_gram_tweet_mle_varied()
print(tweet)

#trigrammi

