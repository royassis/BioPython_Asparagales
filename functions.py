import numpy as np
from Bio import SeqIO
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from datetime import datetime as  dt
from os import listdir, remove
from os.path import isfile, join, getctime

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
