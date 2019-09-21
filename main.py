import numpy as np

from Bio import SeqIO

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from sklearn.svm import SVC
def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
def make_sentence(mySeq,word_size):
    words = getKmers(mySeq, size=word_size)
    sentence = ' '.join(words)
    return sentence


def looping_through(records,arguments, type):
    arr = []
    for record in records:
        str = record.seq._data
        sentence = make_sentence(str, 6)
        if type =="gb":
            label = getattr(record,arguments[0])[arguments[1]][arguments[2]]
        if type == "fasta":
            label = getattr(record, arguments[0])
        arr.append([sentence, label])
    return arr

def seq_to_arr(path,attributes):
    type = path.split(".")[1]
    records = SeqIO.parse(path, type)
    arr = looping_through(records,attributes,type)

    return arr

path = "seq/Asparagales.gb"
arr = seq_to_arr(path,["annotations","taxonomy",-5])
arr = np.array(arr)

#X
cv = CountVectorizer()
X = cv.fit_transform(arr[:,0]).toarray()

pca = PCA(n_components=10)
X = pca.fit_transform(X)


#y
le = LabelEncoder()
y = le.fit_transform(arr[:,1])


clf = SVC(gamma='auto')
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())