import string
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import semcor


stop = stopwords.words('english') + list(string.punctuation) # DA METTERE'``'

def lesk(word,sentence):
    best_sense = get_most_frequent_sense_for_word(word) #è il primo synset preso da wn
    max_overlap = 0
    context = get_senses_for_word(sentence)#frase da cui ho preso il sostantivo tokenizzata
    
    for sense in wn.synsets(word):
           
            signature = compute_signature(sense) #per ogni senso trovato della parola estrai def(gloss) ed examples
            overlap = compute_overlap(signature, context)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense

    print(str(best_sense) + 'o:' + str(max_overlap) + " -> " + str(best_sense.definition()) + str(best_sense.examples()))
    return best_sense


#def senses(word):
#    return wn.synsets(word)

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

def sentence_extraction():
    sentences = (semcor.sents())[:50] # prende le prime 50 frasi da semcor
    semcor_tag_sentences = (semcor.tagged_sents(tag='both'))[:50] # prende le prime 50 frasi da semcor (con tag e semantica)
    equals = 0
    for i,s in enumerate(sentences):
        sentence = ' '.join(s).lower() # unisco le parole per fare frase
        for j,t_s in enumerate(semcor_tag_sentences[i]):
            if type(semcor_tag_sentences[i][j][0]) is Tree and semcor_tag_sentences[i][j][0].label() == 'NN':
                expected_sense = semcor_tag_sentences[i][j].label().synset() # La label è del tipo "Lemma('resignation.n.03.resignation')"
                word = find_suitable_word(semcor_tag_sentences[i][j].leaves()) #levaes estrae la parola sul quale si è visto il pos NN
                break # esce dal ciclo non appena trova la prima parola
        
        print("\n")
        print("Frase da semcor:")
        print(sentence)
        print("Parola da disambiguare:")
        print(word)
        print("Elaborazione lesk:")
        best_sense = lesk(word, sentence)
        if best_sense == expected_sense: #confronto tra il sysnset estratto da semcor e quello trovato da noi con il lesk fatto con wordnet
            equals += 1
    
    print('Accuracy: ' + str(equals/len(sentences)))

def find_suitable_word(words): # lista in input        RIVEDI A CHE SERVE L'IF...
	if len(words) > 1: # se non è una singola parola
		if wn.synsets(''.join(words).lower()) == []: # se le parole unite da un carattere vuoto non ritornano nulla
			return '_'.join(words).lower() # ritorna le parole separate da un _
		return ''.join(words).lower() # ritorna le parole separate da un carattere vuoto
	return ''.join(words).lower() # fa il lower e restituisce una stringa




if __name__ == "__main__":

    sentences= sentence_extraction()
    #sentence = "The table was already booked by someone else."
    #res= lesk(sentence)
   
