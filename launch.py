from flask import Flask, render_template, request, jsonify
from utils import nettoyage, getTrainFromCsv, initVectorizer, predictSentiments
import json
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')

@app.route("/form")
def form():
    return render_template("form.html", title='Form')

@app.route("/result",methods=['POST'])
def result():
    user_text = request.form.get('input_text')
    route='/predict'
    url='http://127.0.0.1:5000'+route
    param={ 'input_text': user_text }
    r=requests.post(url,data=param)
    predicts = r.json()
    print(predicts['pourcentage de fiabilité'])

    return render_template("form.html", title='Prediction', input_text = user_text,
    avis = predicts['Résultat'],  pourcentage = str(predicts['pourcentage de fiabilité']) )


@app.route("/trainingResults", methods=['GET'])
def training_results():
    route='/training'
    url='http://127.0.0.1:5000'+route
    r=requests.get(url)
    resultsTrainings = r.json()

    return render_template("train.html", title='Train', resultsTrainings=
    resultsTrainings['Fiabilité de la machine'] , nbCorpusWords = resultsTrainings['nbCorpusWords'])

@app.route("/predict",methods=['POST'])
def predict():
    user_text = request.form.get('input_text')
    predictResult = predictSentiments(str(user_text))
    if((predictResult[0][0]) == 1.0):
        strAvis = "Avis Positif"
    else:
        strAvis = "Avis Négatif"

    return jsonify({'text_user':user_text, 'Résultat': strAvis, 'pourcentage de fiabilité': (round(predictResult[1], 2) * 100) })

@app.route("/csv")
def training():
    return render_template("csv.html", title='Csv')

@app.route("/training",methods=['GET'])
def train():
    #recupere corpus depuis csv
    Corpus = getTrainFromCsv("corpus.csv")
    #nettoie le corpus 
    Corpus['review_net']=Corpus['review'].apply(nettoyage)
    resultsTraining = initVectorizer(Corpus)
    nbMots = (resultsTraining[1])
    print(nbMots)
    return jsonify({'Fiabilité de la machine': (round(resultsTraining[0], 2) * 100), 'nbCorpusWords':nbMots })

if __name__ == "__main__":
    app.run()