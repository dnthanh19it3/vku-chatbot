# things we need for NLP
import nltk
from flask_cors import CORS
from nltk.stem.lancaster import LancasterStemmer
from underthesea import word_tokenize
from importlib import reload

import train

stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# load our saved model
model.load('./model.tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], np.float64(r[1])))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if(len(i['responses']) > 0):
                            return random.choice(i['responses'])
                        else:
                            return None

            results.pop(0)
    return results

# print(classify('Học phí học kỳ này'))
# print(response('is your shop open today?'))

from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route("/predict")
def predict():
    input_data = request.args.get('msg')
    class_name = classify(input_data)
    response_content = response(input_data)
    print("Content is: ")
    print(type(response(input_data)))
    print("End Content")

    command = 0

    if response_content == "" or response_content is None:
        command = 1

    output_data = {"type" : command, "class" : class_name, "response" : response_content}
    return json.dumps(output_data)
@app.route("/train_app")
def train_app():
    try:
        train.train()
        return json.dumps({"status": "1", "message": "Train thành công!"})
    except Exception as e:
        return json.dumps({"status": "0", "message": "Train thất bại!", "detail": str(e)})
app.run()