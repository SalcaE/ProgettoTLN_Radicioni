import pandas as pd
from nltk.corpus import wordnet as wn
from itertools import product
import math

df = pd.read_csv("Esercizio1\WordSim353\WordSim353.csv")

def max_depth():
    depth = 0
    for ii in wn.all_synsets():
        try:
            depth = max(depth, ii.max_depth())
        except RuntimeError:
            print(ii)
    return depth

def get_hypernyms(synset):
    path = set()
    if not synset.hypernyms():
        return path
    path.update(synset.hypernyms())
    for s in synset.hypernyms():
        path.update(get_hypernyms(s))
    return path

def lcs(s1, s2):
    common_hyp = []
    hyp1 = get_hypernyms(s1)
    hyp2 = get_hypernyms(s2)
    common_hyp = list(hyp1.intersection(hyp2))
    return common_hyp 

def dist_path(syn, root, dist):
    count = 0
    if syn == root or syn.hypernyms() == 0:
        return dist

    for next in syn.hypernyms():
        tmp = dist_path(next, root, dist + 1)
        if (tmp < count or count == 0):
            count = tmp 
    return tmp

def depth(syn):
    depth = 0
    while(len(syn.hypernyms()) != 0 and syn != syn.root_hypernyms()[0]):
        depth += 1
        syn = syn.hypernyms()[0]
    return depth


def wu_palmer(s1, s2):
    lcs_res = lcs(s1, s2)
    depth_lcs = 0

    if len(lcs_res) != 0:
        depth_lcs = depth(lcs_res[0])

    depth_s1 = depth(s1)
    depth_s2 = depth(s2)

    if depth_s1 != 0 and depth_s2 != 0:
        return (2*depth_lcs)/(depth_s1 + depth_s2)
    return 0

def shortest_path(s1, s2):
    intersection = lcs(s1,s2)
    if len(intersection) > 0:
        length = dist_path(s1, intersection[0], 0) + dist_path(s2, intersection[0], 0)
        return 2*max_depth - length
    else:
        return 2*max_depth

def terms_similarity(w1, w2):
    syn1 = wn.synsets(w1)
    syn2 = wn.synsets(w2)
    max = 0
    sim = 0
    for s1 in syn1:
        for s2 in syn2:
            sim = wu_palmer(s1, s2)
            if sim>max:
                max = sim
    return max


max_depth = max_depth()
for i in range(len(df)):
    terms_similarity(df.iloc[i][0], df.iloc[i][1])