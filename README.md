# Mixing Up Sentence Embeddings
#### *Using bag of words embeddings to incorporate prior knowledge into pretrained sentence embeddings*

---

[*Open*](https://gitpod.io/#https://github.com/ryancahildebrandt/priors) *in gitpod*

## *Purpose*
This project is an experiment on the potential usefulness of combining two commonly used sentence/document embedding approaches, bag of words models (tf-idf, hash, count) and pretrained sentence embedding models (RoBERTa, miCSE, sentence-t5). These embeddings are frequently used in text labeling or classification tasks, which depend on information contained in the embedding vectors. Different embedding approaches encode different sorts and amounts of information, and as a result embedding method can drastically change the outcome of a given task. Where neural network based sentence embeddings can more accurately encode the intent, context, and word similarity in an utterance, bag of words models encode specific words explicitly. Traditionally, these methods have not been combined to create a single embedding vector for a variety of reasons, not least of which is the assumption that neural network based embeddings encode all of the information of bag of words models *and then some*. While it's true that neural network embeddings are much more flexible in the types of information they encode and often outperform bag of words models on more complex tasks, that doesn't necessarily mean that bag of words models can't encode useful information for a specific task.
This is where prior knowledge enters the equation and forms the central question for this experiment: 
- **If we have an expectation that for a given problem (text classification in this case), certain words are likely to be informative features in a classification model, is there any benefit in supplementing neural network embeddings with said feature in the form of additional bag of words based embeddings?**


---

## Dataset
The datasets used for the current project were pulled from the following: 
- [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups), for longer documents
- [Bitext Customer Support](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants), for shorter documents

---

## Outputs
- The performance results tables for the [newsgroups](./outputs/pivot_ng.csv) and [bitext](./outputs/pivot_bt.csv) datasets
- The DataPane report, in [html](./outputs/report.html) format and [online](https://cloud.datapane.com/apps/O7vrX27/mixing-up-sentence-embeddings/)
