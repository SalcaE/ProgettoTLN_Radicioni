import pandas as pd
import nltk 
from nltk.lm.preprocessing import padded_everygram_pipeline 
from nltk.lm import MLE
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize


def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from nltk.lm.model.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)




df = pd.read_csv('Esercizio3\\trump_twitter_archive\\tweets.csv')
sentences = list(df['text'].apply(word_tokenize))
#text = sentences.values.tolist()

train, vocab = padded_everygram_pipeline(2, sentences)
lm = MLE(2)
lm.fit(train, vocab)

print(lm.generate(10))
print(generate_sent(lm, num_words=10))
#print(lm.generate(13, random_seed=10))


## meh meh







