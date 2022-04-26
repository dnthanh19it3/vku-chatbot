# things we need for NLP
import nltk
from flask_cors import CORS
from nltk.stem.lancaster import LancasterStemmer
from underthesea import word_tokenize
from importlib import reload
from flask import Flask, request

import getjson
import train

stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle
import json

class Predict:
    json_data = None
    model = None
    data = None
    classes = None
    words = None
    train_x = None
    train_y = None
    net = None
    context = {}
    ERROR_THRESHOLD = 0.25
    intents = None

    init_flag = 0

    def __init__(self):
        app = Flask(__name__)
        app.config["DEBUG"] = True
        CORS(app)
        this = self


        @app.route("/")
        def main():
            if (this.init_flag == 1):
                return json.dumps({"status": "1", "message": "Đã khởi chạy", "detail": "Kiểm tra máy chủ có đang hoạt động hay không"})
            else:
                return json.dumps({"status": "0", "message": "Chưa khởi chạy", "detail": "Kiểm tra máy chủ có đang hoạt động hay không"})

        @app.route("/predict")
        def predict():
            # if(this.init_flag != 1):
            #     try:
            #         self.initPredictEngine()
            #     except Exception as e:
            #         return json.dumps({"status": "0", "message": "Khởi chạy thất bại!", "detail": str(e)})

            input_data = request.args.get('msg')
            class_name = this.classify(input_data)
            response_content = this.response(input_data)
            # print("Content is: ")
            # print(type(this.response(input_data)))
            # print("End Content")

            command = 0

            if response_content == "" or response_content is None:
                command = 1

            output_data = {"type": command, "class": class_name, "response": response_content}
            return json.dumps(output_data)

        @app.route("/train_app")
        def train_app():
            try:
                train.train()
                return json.dumps({"status": "1", "message": "Train thành công!"})
            except Exception as e:
                return json.dumps({"status": "0", "message": "Train thất bại!", "detail": str(e)})

        @app.route("/init")
        def init_predict():
            try:
                self.initPredictEngine()
                return json.dumps({"status": "1", "message": "Khởi chạy thành công!", "detail": "None"})
            except Exception as e:
                return json.dumps({"status": "0", "message": "Khởi chạy thất bại!", "detail": str(e)})
        app.run()

    def initPredictEngine(self):
        try:
            self.data = pickle.load(open("training_data", "rb"))
            self.words = self.data['words']
            self.classes = self.data['classes']
            self.train_x = self.data['train_x']
            self.train_y = self.data['train_y']
            # with open('intents.json', encoding='utf-8') as self.json_data:
            #     self.intents = json.load(self.json_data)

            self.intents = getjson.getJson()

            # Build neural network
            self.net = tflearn.input_data(shape=[None, len(self.train_x[0])])
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, len(self.train_y[0]), activation='softmax')
            self.net = tflearn.regression(self.net)
            # Define model and setup tensorboard
            self.model = tflearn.DNN(self.net, tensorboard_dir='tflearn_logs')
            # load our saved model
            self.model.load('./model.tflearn')
        except Exception as e:
            print("BREAK: " + str(e))
            return 0
            self.init_flag = 1
        return 1

    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    # if show_details:
                        # print ("found in bag: %s" % w)

        return(np.array(bag))



# create a data structure to hold user context

    def classify(self, sentence):
        # generate probabilities from the model
        results = self.model.predict([self.bow(sentence, self.words)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], np.float64(r[1])))
        # return tuple of intent and probability
        print("OK1")
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            # if show_details: print('context:', i['context_set'])
                            self.context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            if(len(i['responses']) > 0):
                                return random.choice(i['responses'])
                            else:
                                return None
                results.pop(0)
        return results


app = Predict()



