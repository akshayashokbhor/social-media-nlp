from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import pkg_resources # -*- coding: utf-8 -*
 




clf = pickle.load(open("sentiment_nlp_model.pkl", 'rb'))
cv=pickle.load(open('BoW.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        message = request.form['message']
        message=[message]
        
        vect = cv.transform(message).toarray()
       
        
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
