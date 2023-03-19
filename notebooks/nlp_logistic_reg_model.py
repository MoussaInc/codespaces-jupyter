# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:26:55 2023

@author: Papis


Training logistic regression model for document classification
dataset: movie_data: 

"""

from pretraitement import tokenizer, tokenizer_porter
import numpy as np
import pandas as pd
import os
from joblib import dump, load # Idem from pickle import dump, load  
#joblib is faster in saving/loading large NumPy arrays,
# whereas pickle is faster with large collections of Python objects


#Packages de scikit learn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer






# Répertoire du dataset movie_data.csv
chemin="C:\\Users\\Papis\\Documents\\Code_python\\NPL\\donnees"

df = pd.read_csv(os.path.join(chemin,"movie_data_cleaned.csv"), encoding='utf-8')

X_train, X_test = df.loc[:25000,'review'].values, df.loc[25000:,'review'].values
y_train, y_test = df.loc[:25000,'sentiment'].values, df.loc[25000:,'sentiment'].values


#nltk.download('stopwords')
#stop = stopwords.words('english')


tfidf = TfidfVectorizer()
vect = tfidf.fit(X_train)
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)
lr_model = LogisticRegression(random_state=10)
lr_model.fit(X_train_vect, y_train)

gs = GridSearchCV(estimator=lr_model, param_grid={'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}, cv=10, verbose=2, n_jobs=1)
gs.fit(X_train_vect, y_train)



with open(os.path.join(chemin, "resultats.txt"), 'a') as file:
    file.write(f'\n La précision du modèle (regression logistique): {lr_model.score(X_test_vect, y_test):.3f}\n')
    file.write('\n-------------------------------------------------------------\n')
    file.write(classification_report(y_test, lr_model.predict(X_test_vect)))
    file.write('\n-------------------------------------------------------------\n')
    file.write(f'La meilleure précision du modèle (GridSearch): {gs.best_estimator_.score(X_test_vect, y_test):.3f}\n')
    file.write(f'Les meilleurs paramètres sont: {gs.best_params_}')
    file.write('\n-------------------------------------------------------------\n')
    file.write(classification_report(y_test, gs.best_estimator_.predict(X_test_vect)))
    
#sauvegarde/persistance du modèle de régression logistic
dump(lr_model, os.path.join(chemin, "lr_model.joblib")) 


