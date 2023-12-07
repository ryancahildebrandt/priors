#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 08:40:38 PM EST 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import random
from itertools import compress
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from sklearn.ensemble import RandomForestClassifier
import random
random.seed(42)

# neural networks
def classifier_dense(emb_dim, label_dim):
    nn = tf.keras.Sequential([
        layers.Input(shape = (emb_dim)),
        layers.Dense(512, activation = "relu"),
        layers.Dense(label_dim, activation = "softmax")
        ])
    
    nn.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = "adam", 
        metrics = ["accuracy"]
        )

    return(nn)

def classifier_conv(emb_dim, label_dim):
    nn = tf.keras.Sequential([
        layers.Input(shape = (emb_dim)),
        layers.Dense(512, activation = "relu"),
        
        layers.Reshape((32,16)),
        layers.Conv1D(filters = 128, kernel_size = 3, activation = "relu"),
        layers.MaxPool1D(2), 
        layers.Flatten(),  
        
        layers.Dense(label_dim, activation = "softmax")
        ])
    
    nn.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = "adam", 
        metrics = ["accuracy"]
        )

    return(nn)

# sk classifiers
def classifier_rf(n_estimators = 200):
    rf = RandomForestClassifier(n_estimators = n_estimators, verbose = 1, n_jobs = -1)

    return(rf)

class Classifier:
    def __init__(self, Dataset_object):
        self.dataset = Dataset_object
        self.cl_dense = classifier_dense(self.dataset.emb_dim, self.dataset.label_dim)
        self.cl_conv = classifier_conv(self.dataset.emb_dim, self.dataset.label_dim)
        self.cl_rf = classifier_rf()
    
    def fit_classifiers(self):
        print("Training Dense Classifier")
        self.cl_dense.fit(x = self.dataset.emb_train, y = self.dataset.y_train, validation_data = (self.dataset.emb_test, self.dataset.y_test), epochs = 20, verbose = 1)
        print("Training Convolutional Classifier")
        self.cl_conv.fit(x = self.dataset.emb_train, y = self.dataset.y_train, validation_data = (self.dataset.emb_test, self.dataset.y_test), epochs = 20, verbose = 1)
        print("Training Random Forest Classifier")
        self.cl_rf.fit(self.dataset.emb_train, self.dataset.y_train)

    def get_results(self):
        self.results = {
            "cl_rf" : self.cl_rf.score(self.dataset.emb_test, self.dataset.y_test),
            "cl_dense" : self.cl_dense.evaluate(x = self.dataset.emb_test, y = self.dataset.y_test)[1],
            "cl_conv" : self.cl_conv.evaluate(x = self.dataset.emb_test, y = self.dataset.y_test)[1],
        }

    def get_misses(self):
        cl_rf_filter = self.cl_rf.predict(self.dataset.emb_test) != self.dataset.y_test
        cl_dense_filter = self.cl_dense.predict(self.dataset.emb_test) != self.dataset.y_test
        cl_conv_filter = self.cl_conv.predict(self.dataset.emb_test) != self.dataset.y_test
        self.misses = {
            "cl_rf" : list(compress(self.dataset.x_test, cl_rf_filter)),
            "cl_dense" : list(compress(self.dataset.x_test, cl_rf_filter)),
            "cl_conv" : list(compress(self.dataset.x_test, cl_rf_filter)),
        }

class Evaluator:
    def __init__(self, Embedder_object):
        self.embedder = Embedder_object
        self.dataset_str_list = self.embedder.ds_list
        self.dataset_obj_list = [self.embedder.ds_st, self.embedder.ds_tfidf, self.embedder.ds_count, self.embedder.ds_rand, self.embedder.ds_zeros]
        self.results = {}
        self.misses = {}
    
    def eval_ds(self):
        for ds,n in zip(self.dataset_obj_list, self.dataset_str_list):
            print(n)
            c = Classifier(ds)
            c.fit_classifiers()
            c.get_results()
            c.get_misses()
            self.results[n] = c.results
            self.misses[n] = c.misses



