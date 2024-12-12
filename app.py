from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import numpy as np
import random
import pickle
import json
import nltk

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://localhost:5173"}})

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load pre-trained model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

fallback_response="I'm sorry I don't quite understand that. Is there anything else I can help you with?"

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    #Get a response based on the predicted intent
    if not ints:
        return fallback_response
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return fallback_response

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if not ints:
        return fallback_response
    res = getResponse(ints, intents)
    return res

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get('message')
    response = chatbot_response(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)