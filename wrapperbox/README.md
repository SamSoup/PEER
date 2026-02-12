# Implementation of [WrapperBox](https://aclanthology.org/2024.blackboxnlp-1.33/), Su et al., 2024

## Step 1: Obtain Vector embeddings

We do this by implementing a SentenceEncoder, and use that on datasets to compute vector embeddings.

See SentenceEncoder.py and `generate_embeddings.py`

## Step 2: Run through white box statistical models 

We adopt the same white box classifiers that can be transferred to regression
(except LMeans, which is designed for classification only.)

kNN, Decision Tree, and Random Forest