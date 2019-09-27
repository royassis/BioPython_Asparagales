# ML_DNA_seq

## Description
A machine learning exercise for predicting species taxa(taxonomy groups) based on input sequences 
with python using the sklearn package. 
 
### Machine Learning:
The M.L part is broken down to segments

#### Getting the data
Multiple gb files of the 5.8 rRNA locus from different species
of the [Asparagales order](https://en.wikipedia.org/wiki/Asparagales) were selected from [the genebank site](https://www.ncbi.nlm.nih.gov/genbank/), 
that contains a annotated sequences database. 

#### Normalisation
Due to the fact that the sequences are of varying length a normalization was performed:

1. The sequences were broken into tokens.
2. They were converted in N length vectors using CountVectorizer function from the 
[sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html/)

Label data (taxonomy group) of the species was taken from the gb files and converted to ordinal values
with sklearn [label encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

#### Building the model 
The predictor that was chosen is SVN  
(https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn-svm-svc)

A pipeline was created that include the following, in that order:
1. Sequence normalisation (as was mentioned)
2. Dimensional reduction (Due to the great length of the sequences) using 
[PCA](https://www.google.com/search?q=pca+sklearn&oq=pca+skl&aqs=chrome.0.0j69i57j0l3j69i60.1735j0j9&sourceid=chrome&ie=UTF-8)
3. SVN

#### hyper-parameter optimisation 
The pipeline parameters (only the PCA number of output dimensions) were optimised using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)	

#### Model persistence 
The model and the ordinal data encoder were saved to binary file using [pickle](https://docs.python.org/3/library/pickle.html#module-pickle)

## Important files in project

1. *train.py* - Python file that have the code for generating the actual model
2. *predict.py :* Python file for using the model and doing predictions
   1. input : gb file (the files that contains the sequences)
   2. output : file with predictions
3. functions.py - general functions
4. model_transformation.joblib - pickled model
5. encoder.joblib - pickled encoder

## How to use
**run in cmd:** predict.py "filename" 

filename must be a .gb file 

outputs a file names "pred.txt" with predictions
