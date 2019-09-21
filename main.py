from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from functions import *

#input params
infile = "seq/Asparagales.gb"
outfile = "svn.txt"
log_folder ="logs"
n=1
m=3

tokens_labels = genebank_to_numpyarr(infile)

#Xy
X = tokens_labels[:, 0]
le = LabelEncoder()
y = le.fit_transform(tokens_labels[:, 1])

delta_scores_i=[]
for i in range (n,m):
    delta, scores = model(X, y,i)
    delta_scores_i.append([i, delta, scores])
    print(i)


