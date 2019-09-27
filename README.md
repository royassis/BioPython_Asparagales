# ML_DNA_seq

## Description
A machine learing exercise for predicting species taxa(taxonomy groups) based on input sequences <br /> with python using the sklearn package. <br />
 
### Learning:
The learning part is broken down to sagments

#### Raw data
Multiple gb files from genebank of the 5.8 rRNA locus from different species
from the Asparagales order were selected from the genebank site (https://www.ncbi.nlm.nih.gov/genbank/)

#### Normalisation

Due to the fact that the sequences are of varying length a normalization was performed:

1. the sequences were broken into tokens 
2. they were converted in N length vectors using CountVectorizer function from the 
sklearn package  <br />
(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
)

Label data (i.e name of taxa Order) of the species was taken from the gb files and converted to ordinal values
with sklearn label encoder. <br />
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

#### Model 


The predictor that was chosen is SVN  <br />
(https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn-svm-svc)

A pipeline was created that include the following, in that order:
1. Sequence normalisation (as was mentioned)
2. Dimensional reduction (Due to the great length of the sequences) using PCA   <br />
(https://www.google.com/search?q=pca+sklearn&oq=pca+skl&aqs=chrome.0.0j69i57j0l3j69i60.1735j0j9&sourceid=chrome&ie=UTF-8)
3. SVN

#### hyperparameter optimisation 
The pipeline parameters (only the PCA number of output dimentions) were optimised using GridSearchCV  <br />
(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)	


#### Model persistance 
The model and the ordinal data encoder were saved to binary file using   <br />
(https://docs.python.org/3/library/pickle.html#module-pickle)



## Important files in project

* train.py - Python file that have the code for generating the actual model
* predict.py - Python file the gets as input a  <br />1. gb file (that contains the sequenses) <br />2. and outputs a list out predictions to file
* functions.py - general function
* model_transformation.joblib - pickled model
* encoder.joblib - pickled encoder


## How to use
run predict.py "filename" <br />
filename must be a .gb file


















