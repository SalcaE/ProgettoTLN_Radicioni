import hashlib
import random
from random import randint
from random import seed
from nltk.corpus import framenet as fn
from nltk.wsd import lesk
import csv
import pandas as pd

def print_frames_with_IDs():
    for x in fn.frames():
        print('{}\t{}'.format(x.ID, x.name))

def get_frams_IDs():
    return [f.ID for f in fn.frames()]   

def getFrameSetForStudent(surname, list_len=5):
    nof_frames = len(fn.frames())
    base_idx = (abs(int(hashlib.sha512(surname.encode('utf-8')).hexdigest(), 16)) % nof_frames)
    print('\nstudent: ' + surname)
    framenet_IDs = get_frams_IDs()
    i = 0
    offset = 0 
    seed(1)
    while i < list_len:
        fID = framenet_IDs[(base_idx+offset)%nof_frames]
        f = fn.frame(fID)
        fNAME = f.name
        print('\tID: {a:4d}\tframe: {framename}'.format(a=fID, framename=fNAME))
        offset = randint(0, nof_frames)
        i += 1


getFrameSetForStudent('Alesandro Spanu')
getFrameSetForStudent('Eduard Salca')
#print_frames_with_IDs()

#print(fn.frame_by_name('Deciding').lexUnit)

ciao=[]
ciao.append(fn.frame_by_id(1604))
ciao.append(fn.frame_by_id(1916))
ciao.append(fn.frame_by_id(2284        ))
ciao.append(fn.frame_by_id(221        ))
ciao.append(fn.frame_by_id(2131        ))


ciao.append(fn.frame_by_id(2046        ))
ciao.append(fn.frame_by_id(269        ))
ciao.append(fn.frame_by_id(2940        ))
ciao.append(fn.frame_by_id(2430        ))
ciao.append(fn.frame_by_id(1919        ))





data = { 
    'nome':[],
    'synset':[],
    'tipo':[]
}


with open('ProgettoTLN_Radicioni/Esercizio2/dati.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for e in ciao:
        #print(i)
        #print(lesk(e.definition,i))
        #print(e._type)
        #data['nome'].append(i)
        #data['synset'].append(lesk(e.definition,i))
        #data['tipo'].append(e._type)
        

        dict= e.FE
       
        main_syn= lesk(e.definition,e.name)
       
       
        fn_type= e._type

        e.ID
        

        writer.writerow([e.name,main_syn,fn_type,e.ID])

       

        for i, a in dict.items():
           
            writer.writerow([i,lesk(a.definition,i),a._type,a.ID])
        
        dict_lu = e.lexUnit
        for i, a in dict_lu.items():
            writer.writerow([i,lesk(a.definition,i.split('.')[0]),a._type,a.ID]) 

#df = pd.read_csv('ProgettoTLN_Radicioni\\Esercizio2\\datiModSapo.csv')  
#print(df.to_string())

#pd.DataFrame(data).to_csv('ProgettoTLN_Radicioni/Esercizio2/dati.csv') '''


