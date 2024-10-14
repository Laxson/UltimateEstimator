import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.preprocessing import StandardScaler
import string 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron


class UltimateEstimator:
    
    def _init_(self, ultimateEstimator):
        self.ultimateEstimator = ultimateEstimator
    
    
    
    #numerical_ultimate_prediction can intake a set of numerical x_train, x_test, and x_open to predict the y_open values
    #when used in terms of closed claims for x_train and x_test this will produce an estimation of the predicted ultimate value of y_open
    
    #taking x_train of numerical values to apply a nueral net on
    #taking x_test of numerical values to test the numerical model before predicting open claims
    #taking x_predict of numerical values to use after model is created
    #taking y_train to train the model on after building with x_train
    #model parameters can also be editable, with learning rate as a suggestion of the model errors out
    def numerical_ultimate_prediction(x_train, x_test, x_open, y_train, y_test, 
                                    n_hidden=3, n_neurons=30, learning_rate=1e-8,
                                    epochs=300):

        #scaling all numerical data in x (independent variables)
        #scaling avoid unintentionally biases and weights between different columns
        scaler = StandardScaler()  
        x_train = scaler.fit_transform(x_train)
        x_open = scaler.transform(x_open)

        #setting the framework for a basic nueral network
        #current parameters are set to service most claims in the range of $1,000 to $10,000,000
        def build_model(n_hidden=n_hidden, n_neurons=n_neurons, learning_rate=learning_rate, input_shape=[x_train.shape[1]]):
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=input_shape))
            for layer in range(n_hidden):
                model.add(keras.layers.Dense(n_neurons, activation="relu"))
            model.add(keras.layers.Dense(1))
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
            model.compile(loss="mse", optimizer=optimizer)
            return model

        #creating the regressor which takes numerical independent variables to get estimate the incurred dollars
        keras_reg = KerasRegressor(build_model)

        #running the regressor and training the nueral network
        keras_reg.fit(x_train, y_train, epochs=epochs, 
                     validation_data =(x_test, y_test),  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

        #finally, production of estimated ultimate dollars on currently open claims
        y_prediction = keras_reg.predict(x_open)

        return y_prediction
    
    
    #text_ultimate_prediction can intake a set of single column text x_train and x_open to predict the y_open values
    #when used in terms of closed claims for x_train this will produce an estimation of the predicted ultimate value of y_open
    
    #taking x_train of text values to apply a nueral net on
    #taking x_open of numerical values to use after model is created
    #taking y_train to train the model on after building with x_train
    #max_iter is the only perceptron parameter that should be edited, higher numbers would lead to slower processing but allow more training of the model
    def text_ultimate_prediction(x_train, x_open, y_train, max_iter=500):

        #creating a short function to standardize most input English text into a singular format of output vector
        def clean_text(text):
            #clean_text is a blank list that will host the text of a single cell                    
            clean_text = []
            #start by taking the input text and sending all letters to lowercase
                #this is presuming no proper nouns in the text will have the same spelling as another English word
            text = text.lower()
            #word_tokenize sends the string of words into a vector/list of words                     
            text = word_tokenize(text)
            for token in text:  
                #the two following if statements clear the current list of words of any punctuation and low-impact English words    
                if token not in list(string.punctuation):
                    if token not in stopwords.words('english'):
                        #once the token has been deemed as valuable to the estimation of incurred dollars it is added to the clean_text for later use         
                        clean_text.append(token)
            #returns a list of vectored lists containing tokens for each word
            return clean_text

        #dictionary_setup will turn the clean_text from about into a dictionary that the perceptron is capable of utilizing
        def dictionary_setup(claim_descriptions):
            #clean_text is a blank list that will hold the created dictionary
            item = []
            #splitting each claim/row into a seperate item
            for claim_description in claim_descriptions:
                #creating a dictionary for each claim description
                dic = {}
                #for loop to create a dictionary entry for each token/word in the claim description
                for token in claim_description:
                    dic[token] = True
                #finally, add the word's dictionary token into the dicctionary
                item.append(dic)
            #returns a list of dictionaries that contain True tokens    
            return item

        #Apply the two above sub-functions to the x_train dataframe                       
        train_description_tokenized = x_train.apply(clean_text)  
        train_description_feature = dictionary_setup(train_description_tokenized)

        #Apply the two above sub-functions to the x_open dataframe                       
        open_description_tokenized = x_open.apply(clean_text)  
        open_description_feature = dictionary_setup(train_description_tokenized)

        #turning training data  and open claim text descriptions into vectors to feed into the perceptrons                         
        vectorizer = DictVectorizer(sparse=True)
        x_train_vec = vectorizer.fit_transform(train_description_feature)
        x_open_vec = vectorizer.transform(open_description_feature)

        #creating the perceptron model that takes in vectors and outputs estimated incurred dollars                          
        perceptron_model = Perceptron(max_iter=max_iter)
        perceptron_model.fit(x_train_vec, y_train)                   

        #finally, production of estimated ultimate dollars on currently open claims
        y_prediction = perceptron_model.predict(x_open_vec)

        return y_prediction                            