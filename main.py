from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from functions import *

#input params
infile = "seq/Asparagales.gb"
outfile = "svn.txt"
n=1
m=3

tokens_labels = genebank_to_numpyarr(infile)

#Xy
cv = CountVectorizer()
X = cv.fit_transform(tokens_labels[:, 0]).toarray()
le = LabelEncoder()
y = le.fit_transform(tokens_labels[:, 1])
clf = SVC(gamma='auto')

delta_scores_i=[]
for i in range (n,m):
    delta, scores = main_func(i,X,y,clf)
    delta_scores_i.append([i, delta, scores])
    print(i)

write_to_file(outfile, delta_scores_i)
