
import numpy as np
import pandas as pd
import tarfile
import pyprind
import os
import re


#Package de scikit learn
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords


# Répertoire données à traiter
chemin="C:\\Users\\Papis\\Documents\\Code_python\\NPL\\donnees"


def extractfile(url):

    """
    objectif: extraction/décompression du fichier zip

    url: le chemin d'accès (absolu) ou répertoire dans lequel le fichier zip est sauvegardé

    """

    if (url==None):
        url = chemin
    
    with tarfile.open(os.path.join(chemin,"aclImdb_v1.tar.gz"), "r:gz") as tar:
        tar.extractall(path=chemin)



def data_csv(url):

    """
    objectif: lecture de chaque fichier contenu dans le dossier décompresser précedemment
             et création du jeux de données (dataset)

    url: le chemin d'accès (absolu) ou répertoire contenant les fichiers décompréssés

    Return: dataset au format csv (dataframe)

    """

    if (url==None):
        url = chemin

    url_1 = os.path.join(url, "aclImdb")
    labels = {"pos": 1, "neg": 0}
    
    # Initialisation de l'object pbar, avec 50 000 itérations (nbr de docs ou d'avis à lire)
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    
    for s in {"train", "test"}:
        for l in {"pos", "neg"}:
                  rep = os.path.join(url_1, s, l)
                  for file in sorted(os.listdir(rep)):
                      with open(os.path.join(rep, file), 'r', encoding='utf-8') as infile:
                          text = infile.read()
                          df = df.append([[text, labels[l]]], ignore_index=True)
                          pbar.update()
    
    df.columns = ["review", "sentiment"]
    # Permutation ou ré-indexation des lignes étant donné que les avis sont initialement triés
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(os.path.join(url, "movie_data.csv"), index=False, encoding='utf-8')

    return df



def text_data_cleaner(text):
    
    """
    objectif: nettoyer un text en supprimant le code html, certaines caractéres,
                ou ponctualisation / signes, etc...
    
    text: text à nettoyer
    
    return: text après nettoyage
    
    """
    # Removing all html markup from the text
    text = re.sub('<[^>]*', ' ', text)
    # Regex to find emoticons which we temporarily stored as emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # Removing all non-word characters from the text via '[\W]+' and convert the text into lowercase characters
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ' ')
    #tokenized = [w for w in text.split() if w not in stop]
 
    return text

def tokenizer(text):
    """
    objectif: split du text en élément singulier (découpage mot par mot)
    
    text: text à découper (via split(" "))
    
    return: text traité sous forme de list

    """
    
    text = text_data_cleaner(text)
    return text.split()


def tokenizer_porter(text):
    """
    Idem que la fonction précédement, sauf que celle-ci utilise l'algorithme Porter stemming
    est déjà implémenté dans le package nltk (Natural Language Toolkit ) de scikit learn

    """
    text = text_data_cleaner(text)
    porter = PorterStemmer()
    
    return [porter.stem(word) for word in text.split()]
    
        
def stop_word_remove(text):
    """
    Suppression des mots trés commun, sousvent courts, qui n'apporte pas
    vraiment d'informations
    Exple: 'and', 'a', 'is', 'has', 'like', etc..
    
    Une liste ou un ensemble de 127 stop_word (en anglais) est déja fournie par nltk

    """
    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    return [w for w in tokenizer_porter(text) if w not in stop]



def data_cleaned(mon_fichier_csv):
    
    """
    Fonction réalisant tout le préprocessing du dataframe movie_data:
                cleaning, tokenizing, removing stop word
                
    Return : None
             Le dataframe nettoyer et sauvegardé sous format csv
    
    """
    
    df = pd.read_csv(os.path.join(chemin, mon_fichier_csv), encoding='utf-8')
    df['review'] = df['review'].apply(stop_word_remove)
    
    # Permutation ou ré-indexation des lignes étant donné que les avis sont initialement triés
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(os.path.join(chemin,"movie_data_cleaned.csv"), index=False, encoding='utf-8')
    