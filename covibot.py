from flask import Flask, render_template, request
import re
import random
import string
import pandas as pd
import datetime
import requests

import nltk
import numpy as np

from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config.update(DEBUG=True)

# Seulement la première fois
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

# Import and text cleaning
f = open('infos_corona.txt', 'r', errors= 'ignore', encoding= 'utf-8')
text = f.read()

# Tokenisation en phrases et mots
phrases_token = nltk.sent_tokenize(text, language = "french")

# Suppression des questions
for p in reversed(range(len(phrases_token))) :
    if phrases_token[p][-1] == "?":
        del phrases_token[p]
        
# Suppression des doublons
phrases_token = list(set(phrases_token)) 

# Text reading in lowercase
# First cleaning
def cleaning(text) :
    text = text.lower()
    text =  re.sub(r'\ufeff', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'(n\.c\.a\.)', 'nca', text )
    text = re.sub(r'covid-19|sars-covid', 'coronavirus', text)
    text = re.sub(r'coronavirus coronavirus', 'coronavirus', text)
    return text
    
phrases_nettoyees = []
for i in range(len(phrases_token)):
    phrases_nettoyees.append(cleaning(phrases_token[i]))

from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()
def StemToken(tokens):
    return [stemmer.stem(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
    return StemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# on définit la fonction qu'on appellera dans le chatbot : elle renvoie 
# la phrase la plus proche de celle posée par l'utilisateur

def rep_covibot(question) :
    
    # Supp des stopwords
    fr_stopwords = get_stop_words('french')
    
    #  Filtre pour la Matrice TF-IDF
    Tfidfvec = TfidfVectorizer(tokenizer = StemNormalize, stop_words= fr_stopwords)
    tf_idf_chat = Tfidfvec.fit(phrases_nettoyees)
    
    # on crée la matrice TF-IDF
    phrases_tf = tf_idf_chat.transform(phrases_nettoyees)
    
    # On a besoin de passer la chaîne de caractère dans une liste :
    question = [question]
    # On calcule les valuers TF-IDF pour la phrase de l'utilisateur
    question_tf = tf_idf_chat.transform(question)
    
    # On calcule la similarité entre la question posée par l'utilisateur
    # et l'ensemble des phrases de la page
    similarity = cosine_similarity(question_tf, phrases_tf).flatten()
    
    # on sort l'index de la phrase étant la plus similaire
    index_max_sim = np.argmax(similarity)
    # Si la similarité max ets égale à 0 == pas de correspondance trouvée
    if similarity[index_max_sim] == 0 :
        bot_resp = "Je n'ai pas trouvé de réponse à votre question, désolé"
    else :
        bot_resp = phrases_token[index_max_sim]
    
    return bot_resp

#### API Nb de cas dans un pays
nb_cas = r"cas en .*?|cas au .*?"

@app.route('/')
def home():    
    return render_template('index.html') 

@app.route('/', methods = ['POST'])
def get_bot_response():

    question = request.form['msg']
        
    if (question == 'quitter'):
        bot_rep = 'Au revoir ! >>> RESTEZ CHEZ VOUS <<<'
        
    # Afficher les infos d'un pays via API
    elif re.fullmatch(nb_cas, question) :
        
        question = re.sub(f"[{string.punctuation}]", " ", question)
        # On récupère le pays renseigné par user
        country = question.split()[-1]
        # On fait une requête
        response = requests.get(f'https://coronavirus-19-api.herokuapp.com/countries/{country}')
        rep = response.json()
        cas = rep['cases']
        todaysCase = rep['todayCases']
        dead = rep['deaths']
        todayDead = rep['todayDeaths']
        guerison = rep['recovered']
        posit = rep['active']
        critic =rep['critical']
        bot_rep = f"Dans les dernières 24h, l'état en {country} est le suivant : {cas} cas recensés depuis le début, {todayDead} nouveaux cas durant les dernières 24h, un total de {dead} morts, {todayDead} morts durant les dernières 24h, {guerison} cas guéris, posit personnes positifs et {critic} cas critiques."
    
    elif question == 'ajouter' :
        question = request.form['msg']
        if question != 'annuler' :
            phrases_token.append(question)
            bot_rep = 'Merci pour votre participation'
              
    else:
        bot_rep = rep_covibot(question)

    return render_template('/index.html', question = question, reponse = bot_rep), print(phrases_token)

if __name__ == "__main__":    
    app.run()