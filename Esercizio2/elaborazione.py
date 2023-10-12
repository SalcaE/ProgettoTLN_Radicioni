import pandas as pd
import spacy
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn

'''def frame_elaboration(row):
    print("print frame",row)

def fe_elaboration(row):
    print("print fe")

def lu_elaboration(row):
    print("print lu")'''

def ctx_frame(frame):
    definition = frame.definition
     
    sentence= ''
    for key, fe in frame.FE.items():
        sentence += ' ' + fe.definition
    
    return elaboration_def(definition+sentence)



def ctx_fe(fe_element):

    definition = fe_element.definition
    
    return elaboration_def(definition)

def ctx_lu(lu_element):
    definition = lu_element.definition
    exemplars = lu_element.exemplars

    sentence=''
    for ex in exemplars:
        sentence += ' ' + ex.annotationSet[0].text

    return elaboration_def(definition+sentence)


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

    doc = sp(definition)
   
    tokens = [token.lemma_.lower() for token in doc     #tokenizzo le frasi
                        if not token.is_stop and 
                        not token.is_punct and token.text.strip()]
    return set(tokens)

def calculate_score(syn,gold,score):
    
    if gold[1] and syn and str(syn) == gold[1]:
        score = score+1
        return score
   
    return score
  

def getData():
    data = pd.read_csv("ProgettoTLN_Radicioni\\Esercizio2\\finale2.csv")
    #print(data)
    



    ciaone=data[data['ID']== 8827]

    frames=[1604,1916,2284,221,2131,2046,269,2940,2430,1919]
    
    #data.to_csv('ProgettoTLN_Radicioni\Esercizio2\csvProva.csv')
   # ciaone = pd.read_csv("ProgettoTLN_Radicioni\Esercizio2\csvProva.csv")
    
    #for index, row in data.iterrows():
        #print('indice ',index)
        #print('row ',row['Type'])
        #typeRow = row['Type']
         
        #loll= eval(typeRow+'_elaboration' + "(row)")
    score = 0
    tot = 0
    for frame_id in frames:
        frame= fn.frame_by_id(frame_id)
        
        context_frame = ctx_frame(frame)
        tot = tot+1
        res_syn_frame= bag_of_words(frame.name,context_frame)

        syn_gold_frame=data[data['ID']== frame.ID]['Syn']
        score = calculate_score(res_syn_frame,syn_gold_frame,score)
        
        for key,el in frame.FE.items():
            context_fe = ctx_fe(el)
            tot = tot+1
            res_syn_fe= bag_of_words(el.name,context_fe)

            syn_gold=data[data['ID']== el.ID]['Syn']
            
            
            score = calculate_score(res_syn_fe,syn_gold,score)
        


        for key,el in frame.lexUnit.items():
            context_lu = ctx_lu(el)
            tot = tot+1
            res_syn_fe= bag_of_words(el.name,context_lu)

            syn_gold=data[data['ID']== el.ID]['Syn']
            
            score = calculate_score(res_syn_fe,syn_gold,score)
            
      
        print(score/tot)
        
       
   
        
        

    
    
    ciao = 1

    
    
        

def main():
    data= getData()

if __name__ == '__main__':
    main()