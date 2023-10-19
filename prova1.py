import random
import nltk
import pandas as pd

# Assicurati di aver installato e scaricato i dati di NLTK


from nltk.util import bigrams
from nltk.probability import FreqDist, ConditionalFreqDist

df = pd.read_csv('Esercizio3\\trump_twitter_archive\\tweets.csv', sep=',')

tokenized_sent=[]
filtered_tokens=[]
for ind in df.index:
    
    sentences = df.loc[:, 'text'][ind]
    tokenized_sent.append(nltk.word_tokenize(str(sentences))) 
    sapo = [word.lower() for word in tokenized_sent[ind] if word.isalpha()]
    filtered_tokens = filtered_tokens +sapo

# Preprocessa il corpus
#tokens = nltk.word_tokenize("Most of the money raised by the RINO losers of the so-called “Lincoln Project” goes into their own pockets. With what I’ve done on Judges Taxes Regulations Healthcare the Military Vets (Choice!) &amp; protecting our great 2A they should love Trump. Problem is I BEAT THEM ALL!")
#filtered_tokens = [word.lower() for word in tokens if word.isalpha()] #se le parole sono lettere e non simboli strani 

# Estrai i bigrammi
bi_grams = list(bigrams(filtered_tokens))

# Calcola le distribuzioni di frequenza
fdist = FreqDist(filtered_tokens)
cfdist = ConditionalFreqDist(bi_grams)

# Funzione per generare un tweet basato su bi-grammi con MLE
def generate_bi_gram_tweet_mle():
    tweet = []
    current_word = random.choice(filtered_tokens)
    tweet.append(current_word)
    
    while len(tweet) < 9:
        next_word = cfdist[current_word].max()
        tweet.append(next_word)
        current_word = next_word
    
    return ' '.join(tweet)

# Genera un tweet basato su bi-grammi con MLE
tweet = generate_bi_gram_tweet_mle()
print(tweet)