from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import GridSearchCV
import pickle


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

trans = Transformer()
model_transformation = Pipeline([("trans",trans),('CountVectorizer', cv), ("pca", pca), ('svc', clf)])
parameters = {"pca__n_components": [i for i in range(1,100,10)]}
model_transformation = GridSearchCV(model_transformation, parameters, cv=5, verbose=2)
model_transformation.fit(X, y)

print(model_transformation.cv_results_)

#Model persistance
pickle.dump(model_transformation, open( 'model_transformation.joblib', "wb" ))



