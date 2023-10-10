import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import semcor
import numpy as np

stop = stopwords.words('english') + list(string.punctuation) + ["'s", "'", "n't","\"", "``", "'d", "'re", "''","''"]

def lesk(word, sentence):
    best_sense = get_most_frequent_sense_for_word(word) #è il primo synset preso da wn
    max_overlap = 0
    context = get_senses_for_word(sentence)#frase da cui ho preso il sostantivo tokenizzata
    
    for sense in wn.synsets(word):  
        signature = compute_signature(sense) #per ogni senso trovato della parola estrai def(gloss) ed examples
        overlap = compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense

def get_most_frequent_sense_for_word(word):
    if wn.synsets(word):
        return wn.synsets(word)[0]
    else:
        return None

def get_senses_for_word(sentence):
    return {i for i in word_tokenize(sentence) if i not in stop}

def compute_signature(sense):

    signature = set()
    gloss = {i for i in word_tokenize(sense.definition())  if i not in stop} #prendo la def e tokenizzo
    examples = {i for i in word_tokenize(' '.join(sense.examples())) if i not in stop } #prendo gli esempi e li tokenizzo
    signature.update(gloss) # ricorda: set mette le parole non ordinate 
    if examples: #necessario, non sempre ho esempi..
        signature.update(examples)
    return signature

def compute_overlap(signature,context):
    return len(context.intersection(signature))

def normal_run():
    sentences = semcor.sents()[:50] # prende le prime 50 frasi da semcor
    sapo = len(semcor.sents())
    pos_sentences = semcor.tagged_sents(tag='pos')[:50]
    sem_sentences = semcor.tagged_sents(tag='sem')[:50]

    accuracy = 0

    for i,s in enumerate(sentences):
        sentence = ' '.join(s).lower()
        word = [t for t in pos_sentences[i] if t.label() == 'NN'][0]
        gold = [t.label() for t in sem_sentences[i] if t[0] == word.leaves()[0]][0]

        print("\n")
        print("Frase da semcor: ")
        print(sentence)
        print("Parola da disambiguare: ")
        print(word.leaves()[0])
        print("Elaborazione lesk: ")

        best_sense = lesk(word.leaves()[0].lower(), sentence)

        if best_sense and gold and best_sense.lemmas()[0] == gold:
            accuracy += 1
    
    print('Accuracy: ' + str(accuracy/len(sentences)))

def random_run():
    sentences = semcor.sents() # prende le prime 50 frasi da semcor         len(semcor.sents())
    print(len(semcor.sents()))
    indexes = np.random.randint(0, 37176, 50) #37176
    accuracy = 0

    for i in indexes:
        sentence = semcor.sents()[i]
        pos_sentences = semcor.tagged_sents(tag='pos')[i]
        sem_sentences = semcor.tagged_sents(tag='sem')[i]
        sentence = ' '.join(sentence).lower()
        words = [t for t in pos_sentences if t.label() == 'NN']
        word = None
        gold = None
        best_sense=None
       
        for i, elem in enumerate(words):
            if elem:
                tmp = [t.label() for t in sem_sentences if type(t)!= list and t[0] == elem.leaves()[0]]
                if len(tmp) > 0:      
                    gold = tmp[0]
                    word = elem
                    break
       
        if word:
            best_sense = lesk(word.leaves()[0].lower(), sentence)
            
        if  best_sense and gold and str(best_sense.lemmas()[0]) == str(gold):
            accuracy += 1
            
    print('Accuracy: ' + str(accuracy/50))

def main():
    #normal_run()
    random_run()

if __name__ == "__main__":
    main()

   
