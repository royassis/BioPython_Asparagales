from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from functions import *


path = "seq/Asparagales.gb"
arr = genebank_to_numpyarr(path)

#Xy
cv = CountVectorizer()
X = cv.fit_transform(arr[:,0]).toarray()
le = LabelEncoder()
y = le.fit_transform(arr[:, 1])
clf = SVC(gamma='auto')


n=1
m=3
arr2=[]
for i in range (n,m):
    delta, scores = main_func(i,X,y,clf)
    arr2.append([i,delta, scores])
    print(i)

write_to_file("svn.txt", arr2)
