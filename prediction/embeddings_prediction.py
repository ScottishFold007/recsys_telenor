import pandas as pd 

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take first 50
df = dataset.tail(20000)

# TODO: create one hot from url and action

# TODO: create dense matrix from url and action

# TODO: concat embeddings for url and action

# TODO: create column for t and t+1

# TODO: split data set

# TODO: model LSTM

# TODO: Train LSTM on training data predicting next item

# TODO: test LSTM on test data