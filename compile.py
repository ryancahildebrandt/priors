#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 05:24:44 PM EST 2023 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

from models import *
from datasets import *
from classifiers import *
from evaluators import *

tf.config.list_physical_devices('GPU')
random.seed(42)

# newsgroups dataset
ng = fetch_20newsgroups(remove = ("headers", "footers", "quotes"))

# bitext dataset
bt = pd.read_csv("data/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv")[["utterance", "intent"]]
bt["intent"] = bt["intent"].factorize()[0]

ng_emb = Embedder(ng.data, ng.target)
bt_emb = Embedder(bt["utterance"].to_numpy(), bt["intent"].to_numpy())

# run through processing steps
results_ng = {}
results_bt = {}
misses_ng = {}
misses_bt = {}

for m in mdl_list:
    print(m)
    # embeddings
    ng_emb.fetch_st_embeddings(m, "newsgroups")
    ng_emb.concat_bow(vocab_ng)
    # datasets
    ng_emb.generate_ds()
    # classifier evaluators
    ng_ev = Evaluator(ng_emb)
    ng_ev.eval_ds()
    # to results_dict
    results_ng[m] = ng_ev.results
    misses_ng[m] = ng_ev.misses

results_df = pd.concat({i: pd.DataFrame.from_dict(j, "index") for i, j in results_ng.items()}, axis = 0).reset_index().rename(columns = {"level_0":"st_model","level_1":"embedding"}).melt(id_vars = ["st_model", "embedding"], value_vars = ["cl_rf", "cl_dense", "cl_conv"], var_name = "classifier", value_name = "accuracy")
results_df.to_csv("outputs/results_ng.csv", index = False)
misses_df = pd.concat({i: pd.DataFrame.from_dict(j, "index") for i, j in misses_ng.items()}, axis = 0).reset_index().rename(columns = {"level_0":"st_model","level_1":"embedding"}).melt(id_vars = ["st_model", "embedding"], value_vars = ["cl_rf", "cl_dense", "cl_conv"], var_name = "classifier", value_name = "doc").explode("doc")
misses_df.to_csv("outputs/misses_ng.csv", index = False)

for m in mdl_list:
    print(m)
    # embeddings
    bt_emb.fetch_st_embeddings(m, "bitext")
    bt_emb.concat_bow(vocab_bt)
    # datasets
    bt_emb.generate_ds()
    # classifier evaluators
    bt_ev = Evaluator(bt_emb)
    bt_ev.eval_ds()
    # to results_dict
    results_bt[m] = bt_ev.results
    misses_bt[m] = bt_ev.misses

results_df = pd.concat({i: pd.DataFrame.from_dict(j, "index") for i, j in results_bt.items()}, axis = 0).reset_index().rename(columns = {"level_0":"st_model","level_1":"embedding"}).melt(id_vars = ["st_model", "embedding"], value_vars = ["cl_rf", "cl_dense", "cl_conv"], var_name = "classifier", value_name = "accuracy")
results_df.to_csv("outputs/results_bt.csv", index = False)
misses_df = pd.concat({i: pd.DataFrame.from_dict(j, "index") for i, j in misses_bt.items()}, axis = 0).reset_index().rename(columns = {"level_0":"st_model","level_1":"embedding"}).melt(id_vars = ["st_model", "embedding"], value_vars = ["cl_rf", "cl_dense", "cl_conv"], var_name = "classifier", value_name = "doc").explode("doc")
misses_df.to_csv("outputs/misses_bt.csv", index = False)
