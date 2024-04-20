import numpy as np
import nltk
import random
import json
import pickle
import joblib
import os
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

os.chdir(os.path.dirname(os.path.realpath(__file__)))

lemmatizer = WordNetLemmatizer()

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
  results = [[i, r] for i, r in enumerate(res) ]
  results.sort(key=lambda x:x[1], reverse=True)
  return_list=[]
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
    flag = True
    while (flag == True):
      user_response = str(text)
      print("\n\nUSER:",user_response)
      user_response = user_response.lower()
      if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks'):
          flag = False
          return "You are Welcome.."
        else:
          ints = predict_class(user_response)
          res = get_response(ints, intents)
          return res
      else:
          flag = False
          return "GoodBye!"

# while(1):
#   print("BOT: ", process_begin(input()))

app = Flask(__name__,template_folder="templates")

@app.route("/")
def aut():
    return render_template("ChatBot_Frontend.html")

@app.route("/receive-data", methods=["POST"])
def receive_data():
    data = request.get_json()
    print("data: ",data)
    result = process_begin(str(data))
    return str(result)

if __name__ == "__main__":
    app.run()
