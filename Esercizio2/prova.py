from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

print(wn.synsets('Result'))

print(wn.synset('place.n.10').definition())
#print(lesk('arousing curiosity or interest', 'interesting'))