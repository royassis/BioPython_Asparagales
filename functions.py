import numpy as np
from Bio import SeqIO
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from datetime import datetime as  dt


def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def make_sentence(mySeq,word_size):
    words = getKmers(mySeq, size=word_size)
    sentence = ' '.join(words)
    return sentence

def records_to_list(records):
    arr = []
    for record in records:
        str = record.seq._data
        sentence = make_sentence(str, 6)
        taxo = record.annotations["taxonomy"][-5]
        arr.append([sentence,taxo])
    return arr

def genebank_to_numpyarr(path):
    file_type = path.split(".")[1]
    records = SeqIO.parse(path, file_type)
    l = records_to_list(records)
    np_arr = np.asarray(l,dtype='U')
    return np_arr

def write_to_file(file_name, array):
    with open(file_name, 'w') as f:
        f.write("n_components, mean_score, delta_time \n")
        for i in array:
            i =[str(j) for j in i]
            f.write(",".join(i) + "\n")

def timer(func):
   def func_wrapper(i,X,y,clf):
       t1 = dt.now()
       scores =  func(i,X,y,clf)
       t2 = dt.now()
       delta = (t2 - t1).seconds
       return delta, scores
   return func_wrapper

@timer
def main_func(i,X,y,clf):
    pca = PCA(n_components=i)
    X_copy = pca.fit_transform(X)
    scores = cross_val_score(clf, X_copy, y, cv=5).mean()
    return scores
