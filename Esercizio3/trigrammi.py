import random
import nltk
import pandas as pd
import string

# Assicurati di aver installato e scaricato i dati di NLTK


from nltk.util import bigrams,trigrams,ngrams
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
bi_grams = list(ngrams(filtered_tokens,3))

# Calcola le distribuzioni di frequenza
fdist = FreqDist(filtered_tokens)


condition_pairs = (((w0, w1), w2) for w0, w1, w2 in bi_grams)

cfdist =nltk.ConditionalFreqDist(condition_pairs)





# Funzione per generare un tweet basato su bi-grammi con MLE
def generate_bi_gram_tweet_mle_varied():
    tweet = []
   
    
   
    
    while len(tweet) < 9:
        current_word = random.choice(filtered_tokens)
        current_word1 = random.choice(filtered_tokens)

        next_options = list(cfdist[current_word, current_word1].keys())
        next_probabilities = [cfdist[current_word, current_word1].freq(w) for w in next_options]
        
        if next_options and next_probabilities:
            next_word = random.choices(next_options, next_probabilities)
            tweet.append(next_word)
        #current_words = (current_word, next_word)
    
    return tweet

# Genera un tweet basato su bi-grammi con MLE e variazione
tweet = generate_bi_gram_tweet_mle_varied()
print(tweet)

#trigrammi

