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


path = "seq/Asparagales.gb"
type = path.split(".")[1]
arr=[]
for record in SeqIO.parse(path,type):
    str = record.seq._data
    sentence=make_sentence(str,6)
    label = record.annotations["taxonomy"][-5]
    arr.append([sentence,label])

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