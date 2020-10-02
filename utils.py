
import re
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
import pandas as pd


#get les stop words langues FR 
def getStopWords():
    fr = SnowballStemmer('french')
    my_stop_word_list = get_stop_words('french')
    final_stopwords_list = stopwords.words('french')
    s_w=list(set(final_stopwords_list+my_stop_word_list)) #concatene les deux listes de stop words
    s_w=[elem.lower() for elem in s_w]
    return s_w

#fonction de nettoyage d'une chaine de characteres
def nettoyage(string):
    l=[]
    print(string)
    string=unidecode(string.lower()) #enleve acccent et passe la chaine en miniscule
    #Sans ponctuation pour le moment
    string=" ".join(re.findall("[a-zA-Z]+", string))  #joint les tuples dans une chaine avec un separateur, ne l'applique que sur ceux qui respecte le regex
    stopwords = getStopWords()
    for word in string.split():
        if word in stopwords:  #si mot dans les stopwords, il n'est pas validé
            continue
        else:  #si mot pas dans les stopwords alors on l'jaoute à la la liste des mots
            #l.append(fr.stem(word))
            l.append(word)
    return ' '.join(l)  #reforme la string en séparant les mots par des espaces 


#entraine la machine, sauvegarde dans le cls.pkl et renvoie le score de fiabilité
def initVectorizer(Corpus):
    vectorizer = TfidfVectorizer()  #initialise matrice tf idf 
    vectorizer.fit(Corpus['review_net']) #met en forme le vectorizer avec la nvlle colonne cleanée
    X=vectorizer.transform(Corpus['review_net'])
    pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb")) #enregistre les mots dans un fichier pickle
    y=Corpus['label'] 
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    cls=LogisticRegression(max_iter=300).fit(x_train,y_train)

    pickle.dump(cls,open("cls.pkl","wb"))
    nbWords = (len(vectorizer.get_feature_names()))

    return  [(cls.score(x_val,y_val)), nbWords]


#on recupère les données d'entrainement depuis le csv et on les traite afin de savoir si l'avis est negatof ou positif
def getTrainFromCsv(name):
    df=pd.read_csv(name)
    len(df),len(df.drop_duplicates('review'))
    df['l_review']=df['review'].apply(lambda x:len(x.split(' ')))
    df=df[df['l_review']>5]
    df['label']=df['rating'] 
    positif=df[df['label']>3].sample(391) 
    negatif=df[df['label']<3]
    Corpus=pd.concat([positif,negatif],ignore_index=True)[['review','label']] 
    positifs = Corpus[Corpus["label"] > 3].index
    negatifs = Corpus[Corpus["label"] < 3].index
    Corpus.loc[positifs, "label"]  = 1
    Corpus.loc[negatifs, "label"]  = 0
    print(Corpus)
    return Corpus


#évalue la phrase en recupèrant le modele grâce  au fichier feature.pkl
def predictSentiments(phrase):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    user = transformer.fit_transform(loaded_vec.fit_transform([nettoyage(phrase)]))
    cls=pickle.load(open("cls.pkl", "rb"))
    #indicateur de fiabilité
    x = (cls.predict(user),cls.predict_proba(user).max())
    return x
    
     
