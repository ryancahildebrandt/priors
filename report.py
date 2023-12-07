#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 07:27:22 PM EST 2023 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import datapane as dp
import pandas as pd
import numpy as np

from evaluators import *

# results files
results_ng = pd.read_csv("outputs/results_ng.csv")
results_bt = pd.read_csv("outputs/results_bt.csv")

pivot_ng = results_ng.pivot(index = ["st_model", "classifier"], columns = ["embedding"], values = ["accuracy"])
pivot_ng.columns = pivot_ng.columns.get_level_values(1).rename("")
pivot_ng.reset_index(inplace = True)

pivot_bt = results_bt.pivot(index = ["st_model", "classifier"], columns = ["embedding"], values = ["accuracy"])
pivot_bt.columns = pivot_bt.columns.get_level_values(1).rename("")
pivot_bt.reset_index(inplace = True)

results_ng.groupby("embedding").aggregate(func = "mean", numeric_only = True)

for i in ["tfidf", "count"]:
    for j in ["st", "rand", "zeros"]:
        pivot_bt[f"diff_{i}_{j}"] = pivot_bt[f"ds_{i}"] - pivot_bt[f"ds_{j}"]
        pivot_ng[f"diff_{i}_{j}"] = pivot_ng[f"ds_{i}"] - pivot_ng[f"ds_{j}"]

rprt = dp.Report(
    dp.Text("""
# Mixing Up Sentence Embeddings
### *Using bag of words embeddings to incorporate prior knowledge into pretrained sentence embeddings*

---

## Purpose
This project is an experiment on the potential usefulness of combining two commonly used sentence/document embedding approaches, bag of words models (tf-idf, hash, count) and pretrained sentence embedding models (RoBERTa, miCSE, sentence-t5). These embeddings are frequently used in text labeling or classification tasks, which depend on information contained in the embedding vectors. Different embedding approaches encode different sorts and amounts of information, and as a result embedding method can drastically change the outcome of a given task. Where neural network based sentence embeddings can more accurately encode the intent, context, and word similarity in an utterance, bag of words models encode specific words explicitly. Traditionally, these methods have not been combined to create a single embedding vector for a variety of reasons, not least of which is the assumption that neural network based embeddings encode all of the information of bag of words models *and then some*. While it's true that neural network embeddings are much more flexible in the types of information they encode and often outperform bag of words models on more complex tasks, that doesn't necessarily mean that bag of words models can't encode useful information for a specific task.
This is where prior knowledge enters the equation and forms the central question for this experiment: 
- **If we have an expectation that for a given problem (text classification in this case), certain words are likely to be informative features in a classification model, is there any benefit in supplementing neural network embeddings with said feature in the form of additional bag of words based embeddings?**

---

## Approach
To explore this question, the basic approach I took was to compare different combinations of pretrained embeddings and bag of words embeddings by using them to train 3 different classifiers across 2 datasets.

- **Pretrained Models (from HuggingFace.co via SentenceTransformers)**
    - flax-sentence-embeddings/all_datasets_v3_roberta-large
    - sap-ai-research/miCSE
    - sentence-transformers/sentence-t5-base
    - sentence-transformers/all-roberta-large-v1
    - sentence-transformers/all-distilroberta-v1
    - sentence-transformers/all-mpnet-base-v2
    - sentence-transformers/all-MiniLM-L12-v2
    - sentence-transformers/all-MiniLM-L6-v2

- **Bag of Words Models (from scikit-learn)**
    - TF-IDF
    - Count

*For each bag of words model, the vocabulary to embed for each document was based on the specific dataset from which the document came, and included a list of words expected to be relevant to the classification task for that dataset*

- **Classifiers (from tensorflow and scikit-learn)**
    - Dense neural network
    - Convolutional neural network
    - RandomForest classifier

- **Datasets**
    - [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups), for longer documents
    - [Bitext Customer Support](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants), for shorter documents

---

## Results
In addition to comparing the combined bag of words/pretrained embeddings to the pretrained embeddings alone, I included two other baselines to control for the potential benefit of simply having more features. These were included as a vector of the corresponding vocabulary length, populated with either random numbers or zeroes. These are included with the below results, as are some comparisons of accuracy across pretrained embeddings and classifiers.
Across all models and classifiers, accuracy was near 100% for the bitext dataset. As such, the discussion of performance differences will be focused on the newsgroups dataset.

### Models
"""),
dp.Group(
    dp.Table(results_ng.groupby("st_model").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "NewsGroups"),
    dp.Table(results_bt.groupby("st_model").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "Bitext"),
    columns = 2),
dp.Text("""
Averaged across all embeddings and classifiers, mdl_mpnetv2 had the best performance at 73.6%, and mdl_micse had the worst performance at 66.0%. These and the other models are relatively in line with Sentence Transformers' evaluation [results](https://www.sbert.net/docs/pretrained_models.html) 

### Classifiers
"""),
dp.Group(
    dp.Table(results_ng.groupby("classifier").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "NewsGroups"),
    dp.Table(results_bt.groupby("classifier").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "Bitext"),
    columns = 2),
dp.Text("""
The classifiers performed well overall and relatively similarly. The simple dense neural network classifier performed the best at 71.4%, with the convolutions added in the convolutional classifier detracting from the classification accuracy a bit. This isn't necessarily surprising given the traditional uses of convolutions in image and sequence processing. The random forest performed marginally worse than the neural network based classifiers at 69.4%, but was slightly faster trained when using a GPU for the neural network training.

### Embeddings
"""),
dp.Group(
    dp.Table(results_ng.groupby("embedding").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "NewsGroups"),
    dp.Table(results_bt.groupby("embedding").aggregate(func = "mean", numeric_only = True).style.format({"accuracy" : "{:,.2%}".format}), caption = "Bitext"),
    columns = 2),
dp.Text("""
Among the embedding techniques used, pretrained embeddgings combined with tf-idf had the best performance across classifiers and embedding models, coming in at 71.7%, as compared to the baseline pretrained embeddings (71.1%), pretrained + zeros embeddings (71.0%), and the pretrained + random embeddings (67.7%). Pretrained + count embeddings did have a sligt advantage over baseline at (71.3%), but the differences are fairly small overall.

---

## Conclusion
Below are tables for the full performance comparisons across all datasets, models, classifiers, and embeddings. On the right side I've added comparisons of tfidf and count embeddings against each embedding baseline to isolate performance differences between each embedding type for all models and classifiers.

- The largest differences can be seen in the lowest performing embedding models, namely mdl_micse.
- Pretrained + tf-idf and count embeddings performed consistently better than pretrained alone, though for this model the count embeddings performed better than tf-idf
- Tf-idf and count embeddings had a larger advantage over random embeddings than zeros embeddings
- The variation in performance from model to model and embedding to embedding are relatively small across the board, and the differences observed should be taken with a grain of salt

**Overall, the benefits seen here suggest that adding additional embedding features in the form of bag of word embeddings *can* result in some benefit in text classification. While this suggests that the incorporation of prior knowledge into sentence embeddings is possible, it should be noted that the performance gains are likely minimal. Other methods, such as model fine tuning or dimensionality reduction can have a more dramatic impact on classifier accuracy, so the practical applications of these results are questionable.**
"""),
dp.Table(pivot_ng.style.background_gradient(cmap = "YlGn"), caption = "NewsGroups"),
dp.Table(pivot_bt.style.background_gradient(cmap = "YlGn"), caption = "Bitext"),
)

rprt.save(path = "./outputs/report.html", open = False)
rprt.upload(name = "Mixing Up Sentence Embeddings", open = False, publicly_visible = True)
#https://cloud.datapane.com/apps/O7vrX27/mixing-up-sentence-embeddings/