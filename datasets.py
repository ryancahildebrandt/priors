#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 05:07:38 PM EST 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""

# imports
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import random
random.seed(42)

from models import *

ds_list = ["ds_st", "ds_tfidf", "ds_count", "ds_rand", "ds_zeros"]

class Dataset:
    def __init__(self, x, y, embs, split = 0.2):
        self.label_dim = len(set(y))
        self.emb_dim = len(embs[0])
        self.x_train, self.x_test, self.y_train, self.y_test, self.emb_train, self.emb_test = train_test_split(x, y, embs, shuffle = True, test_size = split)

class Embedder:
    def __init__(self, x, y, n_row = None):
        self.x = x[:n_row]
        self.y = y[:n_row]
        self.len = len(self.x)
        self.ds_list = ds_list

    def fetch_st_embeddings(self, st_model_str, dataset_name):
        self.st_model = eval(st_model_str)
        self.mdl_str = st_model_str
        self.dataset_name = dataset_name
        emb_path = f"data/{st_model_str}_{dataset_name}_embeddings.pickle"
        file_exists = os.path.isfile(emb_path)
        
        if file_exists:
            with open(emb_path, "rb") as file:
                self.dict = pickle.load(file)
            print(f"Embedding file exists, loading from {emb_path}")
            self.emb = np.array(self.dict["emb"])
        else:
            print(f"Embedding file does not exist, calculating and writing embeddings to {emb_path}")
            self.emb = self.st_model.encode(self.x, show_progress_bar = True)
            self.dict = {"utt" : self.x, "emb" : self.emb, "target" : self.y}
            with open(emb_path, "wb") as file:
                pickle.dump(self.dict, file, protocol = pickle.HIGHEST_PROTOCOL)

        print("ST embeddings loaded")

    def concat_bow(self, vocab):
        bow_tfidf = TfidfVectorizer(vocabulary = vocab)
        bow_count = CountVectorizer(vocabulary = vocab)
        vocab_len = len(vocab)
        self.emb_tfidf = np.concatenate([self.emb, bow_tfidf.fit_transform(self.x).toarray()], axis = 1)
        self.emb_count = np.concatenate([self.emb, bow_count.fit_transform(self.x).toarray()], axis = 1)
        self.emb_rand = np.concatenate([self.emb, np.random.rand(self.len, vocab_len)], axis = 1)
        self.emb_zeros = np.concatenate([self.emb, np.zeros((self.len, vocab_len))], axis = 1)

        print("BOW embeddings added")

    def generate_ds(self):
        self.ds_st = Dataset(self.x, self.y ,self.emb)
        self.ds_tfidf = Dataset(self.x, self.y ,self.emb_tfidf)
        self.ds_count = Dataset(self.x, self.y ,self.emb_count)
        self.ds_rand = Dataset(self.x, self.y ,self.emb_rand)
        self.ds_zeros = Dataset(self.x, self.y ,self.emb_zeros)
        
        print("Datasets generated")
