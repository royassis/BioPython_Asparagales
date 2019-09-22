import numpy as np
from Bio import SeqIO
from sklearn.model_selection import cross_val_score
from datetime import datetime as  dt
from os import listdir, remove
from os.path import isfile, join, getctime
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin


######NLP functions######
def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def make_sentence(mySeq,word_size):
    words = getKmers(mySeq, size=word_size)
    sentence = ' '.join(words)
    return sentence


######Initial work######
#Process functions#
def process_function_taxonomy(record):
    str = record.seq._data
    sentence = make_sentence(str, 6)
    taxo = record.annotations["taxonomy"][-5]
    return [sentence, taxo]

def process_function_return_string(record):
    str = record.seq._data
    return str

#Main iterations#
def get_features(records,process_function):
    arr = []
    for record in records:
        tmp_arr = process_function(record)
        arr.append(tmp_arr)
    return arr

def genebank_to_numpyarr(path,process_function):
    file_type = path.split(".")[1]
    records = SeqIO.parse(path, file_type)
    l = get_features(records,process_function)
    np_arr = np.asarray(l,dtype='U')
    return np_arr



######Model######
def timer(func):
   def func_wrapper(X,y,i):
       t1 = dt.now()
       scores =  func(X,y,i)
       t2 = dt.now()
       delta = (t2 - t1).seconds
       return delta, scores
   return func_wrapper

@timer
def model(X, y,i):
    clf = SVC(gamma='auto')
    cv =CountVectorizer()
    pca = TruncatedSVD(n_components=i)
    model_transformation = Pipeline([('CountVectorizer', cv), ("pca", pca), ('svc', clf)])
    scores = cross_val_score(model_transformation, X, y, cv=5).mean()
    return scores



######Writing to file######
def write_to_file(out_path, array):
    with open(out_path, 'w') as f:
        f.write("n_components, mean_score, delta_time \n")
        for i in array:
            i =[str(j) for j in i]
            f.write(",".join(i) + "\n")

def check_and_remove_files(file_list, log_folder):
    file_list.sort(key=lambda x: x[0])
    if file_list.__len__() > 5:
        path = log_folder+"/"+file_list[0][1]
        remove(path)

def get_path(log_folder,outfile):
    time = dt.now().strftime("%d_%m_%Y__%H_%M_%S")
    out_path = log_folder + "/" +outfile+ "_"+time
    return out_path

def get_file_list(log_folder):
    file_list= [[getctime(join(log_folder, f)),
          f]
         for f in listdir(log_folder) if isfile(join(log_folder, f))]
    return file_list

def smart_write(log_folder,outfile , array):
    file_list = get_file_list(log_folder)
    check_and_remove_files(file_list, log_folder)
    out_path =get_path(log_folder,outfile)
    write_to_file(out_path, array)


######Classes######
class Transformer(BaseEstimator, TransformerMixin):
    """An example of classifier"""

    def __init__(self, k_mer=6):
        self.k_mer = k_mer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X= [make_sentence(i[0], self.k_mer) for i in X]
        return X



