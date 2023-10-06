import pandas as pd
from nltk.corpus import wordnet as wn
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
    path = []
    if not synset.hypernyms():
        return path
    path = synset.hypernyms()
    for s in synset.hypernyms():
        path = path + get_hypernyms(s)
    return path

def lcs(s1, s2):
    #DEL CASO DA AGGIUNGERE TE STESSO (TIGER / TIGER)
    common_hyp = []
    hyp1 = get_hypernyms(s1)
    hyp2 = get_hypernyms(s2)
    common_hyp = [a for a in list(hyp1) if a in list(hyp2)]
    return common_hyp

def dist_path(syn, root, dist):
    count = 0
    if syn == root or syn.hypernyms() == 0:
        return dist
    for next in syn.hypernyms():
        tmp = dist_path(next, root, dist + 1)
        if (tmp < count or count == 0):
            count = tmp 
    return count

def depth(s):
    depth = 0
    next_hyper = s
    while(next_hyper != s.root_hypernyms()[0] and len(next_hyper.hypernyms())!=0):
        depth += 1
        next_hyper = next_hyper.hypernyms()[0]
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

def leakcock_chodorow(s1, s2):
    intersection = lcs(s1,s2)
    if len(intersection) != 0:
        length = dist_path(s1, intersection[0], 0) + dist_path(s2, intersection[0], 0)
        if length != 0:
            return -math.log(length/(2*max_depth))
        return -math.log(length+1/(2*max_depth+1))
    return 0

def terms_similarity(w1, w2):
    syn1 = wn.synsets(w1)
    syn2 = wn.synsets(w2)
    scores_wu_palmer = []
    scores_shortest_path = []
    scores_leakcock_chodorow = []

    if not syn1 or not syn2:
        return 0,0,0
    
    for s1 in syn1:
        for s2 in syn2:
            scores_wu_palmer.append(wu_palmer(s1, s2))
            scores_shortest_path.append(shortest_path(s1, s2))
            scores_leakcock_chodorow.append(leakcock_chodorow(s1, s2))
    return max(scores_wu_palmer), max(scores_shortest_path), max(scores_leakcock_chodorow)

max_depth = max_depth()

def main ():
    res_wu_palmer = []
    res_shortest_path = []
    res_leakcock_chodorow = []
    for i in range(len(df)):
        scores = terms_similarity(df.iloc[i][0], df.iloc[i][1])
        res_wu_palmer.append(scores[0])
        res_shortest_path.append(scores[1])
        res_leakcock_chodorow.append(scores[2])

    df['wu_palmer'] = res_wu_palmer
    df['shortest_path'] = res_shortest_path
    df['leakcock_chodorow'] = res_leakcock_chodorow

    print(df)
    print()
    print("Wu & Palmer pearson correlation coefficient")
    print(df['Human (mean)'].corr(df['wu_palmer'], method='pearson'))

    print("Wu & Palmer spearman correlation coefficient")
    print(df['Human (mean)'].corr(df['wu_palmer'], method='spearman'))
    print()
    print("Shortest path pearson correlation coefficient")
    print(df['Human (mean)'].corr(df['shortest_path'], method='pearson'))

    print("Shortest path spearman correlation coefficient")
    print(df['Human (mean)'].corr(df['shortest_path'], method='spearman'))
    print()
    print("Leakcock Chodorow pearson correlation coefficient")
    print(df['Human (mean)'].corr(df['leakcock_chodorow'], method='pearson'))

    print("Leakcock Chodorow spearman correlation coefficient")
    print(df['Human (mean)'].corr(df['leakcock_chodorow'], method='spearman'))

if __name__ == "__main__":
    main()
   