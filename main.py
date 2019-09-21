from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import GridSearchCV


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


#Learning
clf = SVC(gamma='auto')
cv = CountVectorizer()
pca = TruncatedSVD(n_components=2)
model_transformation = Pipeline([('CountVectorizer', cv), ("pca", pca), ('svc', clf)])

parameters = {"pca__n_components": [1, 10]}

model_transformation = GridSearchCV(model_transformation, parameters, cv=5)

model_transformation.fit(X, y)


"""
delta_scores_i=[]
for i in range (n,m):
    delta, scores = model(X, y,i)
    delta_scores_i.append([i, delta, scores])
    print(i)
"""

