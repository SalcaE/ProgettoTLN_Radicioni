import pandas as pd
import spacy
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn
import numpy as np
import re

def ctx_frame(frame):
    definition = frame.definition
    sentence= ''
    for key, fe in frame.FE.items():
        sentence += ' ' + fe.definition

    for key, lu in frame.lexUnit.items():
        sentence += ' ' + lu.definition    
    
    return elaboration_def(definition+sentence+frame.name)

def ctx_fe(fe_element):

    definition = fe_element.definition + fe_element.name
    
    return elaboration_def(definition)

def ctx_lu(lu_element):
    definition = lu_element.definition
    exemplars = lu_element.exemplars

    sentence=''
    for ex in exemplars:
        sentence += ' ' + ex.annotationSet[0].text

    return elaboration_def(definition+sentence+lu_element.name)

def ctx_wn(syn):

    syn_def = syn.definition()
    
    example_def = ''
    for ex in syn.examples():
        example_def+= ' '+ex

    hypernyms_def= ''
    for hypernym in syn.hypernyms():
        hypernyms_def+=' ' + hypernym.definition()

    hyponym_def  = ''
    for hyponym  in syn.hyponyms():
        hyponym_def+=' ' + hyponym.definition()
         

    return elaboration_def(syn_def+example_def+hypernyms_def+hyponym_def)

def bag_of_words(element,contex_fe):
    i=0
    syns = wn.synsets(element)
    max_intersection = 0
    best_syn = None
    for syn in syns:
        tmp = len(ctx_wn(syn).intersection(contex_fe)) + 1
        if tmp > max_intersection:
            max_intersection = tmp
            best_syn = syn
    
    return best_syn

def elaboration_def(definition):
    
    sp = spacy.load('en_core_web_md')

    definition = re.sub(r'[^\w\s]','',definition)

    doc = sp(definition)

    tokens = [token.lemma_.lower() for token in doc     #tokenizzo le frasi
                        if not token.is_stop and 
                        not token.is_punct and token.text.strip()]
    
    return set(tokens)

def calculate_score(syn,gold,score):
    syn_gold = gold.values
    print('grafico: ',str(syn),' gold: ',str(syn_gold[0]))
    if syn_gold[0] and syn and str(syn) == str(syn_gold[0]):
        score = score+1
        return score
   
    return score

def path_search(synset_w,syn_el, path, paths):
    tmp = path + [synset_w]  #percorso temporaneo

    if synset_w == syn_el:
        paths.append(tmp)
        return paths
    
    if  len(tmp) >= 3:
        return
    
    for hypernym in synset_w.hypernyms():
        path_search(hypernym,syn_el, tmp, paths)
    for hyponym in synset_w.hyponyms():
        path_search(hyponym,syn_el, tmp, paths)
    
    return paths

def getScore(syn_el, ctx_el):
    res = 0
    for word in ctx_el:   #prendo token definizione
        for syn_w in wn.synsets(word):
           
            paths = path_search(syn_w, syn_el,[],[])
           
            if paths:   #se trova un percorso
                for path in paths:
                    res += np.exp(-len(path)-1)
    return res
  
def approccio_grafico(element, ctx_el):
    best_syn = None
    max_score = 0 
    res = 0
   
    for syn in wn.synsets(element):  #synset sull'elment/frame principale
        sum = 0

        for syn2 in wn.synsets(element): #riscorre..
            sum += getScore(syn2,ctx_el)

        if sum > 0:
            res = getScore(syn, ctx_el) / sum # score(s,w)/score(s',w')
   
        if res > max_score:  #argmax
            max_score = res
            best_syn = syn
            
    return best_syn

def getData(algo_type):
    data = pd.read_csv("Esercizio2\\annotation.csv")
    frames=[1604,1916,2284,221,2131,2046,269,2940,2430,1919]
    
    score = 0
    tot = 0
    
    for frame_id in frames:
        tot = tot+1
        frame= fn.frame_by_id(frame_id)
        context_frame = ctx_frame(frame)
        syn_gold_frame=data[data['ID']== frame.ID]['Syn']

        if algo_type == 'bag_of_words':
            res_syn_frame= bag_of_words(frame.name,context_frame)
        else:
            res_syn_frame=approccio_grafico(frame.name, context_frame)

        score = calculate_score(res_syn_frame,syn_gold_frame,score)
       
        for key,el in frame.FE.items():
            tot = tot+1
            context_fe = ctx_fe(el)
            syn_gold=data[data['ID']== el.ID]['Syn']
            
            if algo_type == 'bag_of_words':
                res_syn_fe= bag_of_words(el.name,context_fe)
            else:
                res_syn_fe=approccio_grafico(key, context_fe)
              
            score = calculate_score(res_syn_fe,syn_gold,score)
        
        for key,el in frame.lexUnit.items():
            tot = tot+1
            context_lu = ctx_lu(el)
            syn_gold=data[data['ID']== el.ID]['Syn']

            if algo_type == 'bag_of_words':
                res_syn_fe= bag_of_words(el.name,context_lu)
            else:
                res_syn_fe=approccio_grafico(key, context_lu)
              
            score = calculate_score(res_syn_fe,syn_gold,score)
              
    return(score/tot)


def main():
    score = getData('cazzo ')
    print(score)

if __name__ == '__main__':
    main()