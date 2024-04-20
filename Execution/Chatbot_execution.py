import streamlit as st
import numpy as np
import nltk
import random
import json
import pickle
import joblib
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = joblib.load('chatbot_model.pkl')

def sentence_cleaning(sentence):
    sen_words = nltk.word_tokenize(sentence)
    sen_words = [lemmatizer.lemmatize(word) for word in sen_words]
    return sen_words

def wordBag(sentence):
    sen_words = sentence_cleaning(sentence)
    bag = [0] * len(words)
    for w in sen_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = wordBag(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THR = 0.25
    results = [[i, r] for i, r in enumerate(res)]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    max_prob_intent = intents_list[0]['intent']
    max_prob = float(intents_list[0]['probability'])
    if max_prob < 0.8:
        return "I'm sorry, I didn't understand your question."
    for intent in intents_list:
        if float(intent['probability']) > max_prob:
            max_prob = float(intent['probability'])
            max_prob_intent = intent['intent']
    tag = max_prob_intent
    list_of_intents = intents_json['tags']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def process_begin(text):
    user_response = str(text).lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            return "You are Welcome.."
        else:
            ints = predict_class(user_response)
            res = get_response(ints, intents)
            return res
    else:
        return "GoodBye!"

def main():
    st.title("Chatbot")

    # Read HTML file content
    with open('templates/Chatbot_frontend.html', 'r') as f:
        html_content = f.read()

    # Display HTML content
    st.components.v1.html(html_content)

    user_input = st.text_input("You: ")
    if user_input:
        response = process_begin(user_input)
        st.write("Bot:", response)

if __name__ == "__main__":
    main()
