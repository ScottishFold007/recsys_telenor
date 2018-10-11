import itertools
import numpy as np
import pandas as pd 
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take last 20 000 items
df = dataset.tail(10)

actions = df['action'].fillna('')
urls = df['url'].fillna('')

'''
text = urls.iloc[0]
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
print(words)
'''
def create_one_hot_vectors(data):
    words = [text_to_word_sequence(e) for e in data.values]

    vocabulary = set(itertools.chain.from_iterable(words))
    vocabulary_size = len(vocabulary)
    increased_size = round(vocabulary_size*1.3)

    # One-hot encodes a text into a list of word indexes of size n. TODO: check if this is ok for converting string to int
    encoded_documents = [one_hot(d,increased_size) for d in data] 

    # find length of longest string
    max_length = len(max(encoded_documents, key=len))

    # pad documents
    padded_documents = pad_sequences(encoded_documents, maxlen=max_length, padding='post')
    print(padded_documents[0])
    print(padded_documents[1])
    for p in padded_documents:
        print(p.shape)
    # create one-hot vectors
    one_hot_list = []
    for d in padded_documents:
            one_hot_list.append(to_categorical(d))
   
    for d in one_hot_list:
        print(d.shape)
    one_hot_vectors = np.array(one_hot_list)

    return increased_size, one_hot_vectors

url_vocabulary_size, url_one_hot_vectors = create_one_hot_vectors(urls)
action_vocabulary_size, action_one_hot_vectors = create_one_hot_vectors(actions)
print('url vocabulary size:',url_vocabulary_size)
print(url_one_hot_vectors.shape)
print(url_one_hot_vectors[0])
#print('actions vocabulary size:',action_vocabulary_size)
# TODO: create dense matrix from url and action

# TODO: concat embeddings for url and action

# TODO: create column for t and t+1

# TODO: split data set

# TODO: model LSTM

# TODO: Train LSTM on training data predicting next item

# TODO: test LSTM on test data

# TODO: split preprocessing data and lstm into different scripts - save and load preprocessed data from file (pickle)