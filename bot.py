import random
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
# nltk.download('popular')

lemmatizer = WordNetLemmatizer()

# Load pre-trained model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

fallback_response="I'm sorry I don't quite understand that. Is there anything else I can help you with?"

def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - array of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if the word is in the vocabulary
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    # Predict the intent class for a given sentence
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    # Get a response based on the predicted intent
    if not ints:
        return fallback_response
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return fallback_response


def chatbot_response(msg):
    # Generate a chatbot response
    ints = predict_class(msg, model)
    if not ints:
        return fallback_response
    res = getResponse(ints, intents)
    return res


# Main loop for terminal interaction
if __name__ == "__main__":
    print("Chatbot is running! (Type 'quit' to exit)\n")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Exiting chatbot. Goodbye!")
            break
        response = chatbot_response(message)
        print(f"Bot: {response}")
