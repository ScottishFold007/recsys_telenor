import pickle
from datetime import datetime as dt
import torch
import random
import numpy as np


def load_data():
    t = pickle.load( open( "data/cleaned_dataset.p", "rb" ) )
    return t


def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def split_dataset(dataset):
    random.shuffle(dataset)
    train_size = int(0.5 * len(dataset))
    train = dataset[0:train_size]
    test = dataset[train_size:-1]
    return train, test

def create_dataset(df):
    
    vocab = list(set(df['action_cleaned']))
        
    input_indices = integer_encode(vocab, df['action_cleaned'])
    df = df.assign(input_index=input_indices)

    x = [] 
    for uuid, row in df.groupby('UUID'):
        x.append(torch.LongTensor(row['input_index'].values))

    # split data into training and testing
    x_train, x_test = split_dataset(x)

    return x_train, x_test, vocab    


def create_embeddings(vocab):
    w2v = pickle.load( open( "data/w2v.p", "rb" ) )
    weights_matrix = np.zeros((len(vocab), 20))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = w2v[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(20, ))
    return weights_matrix

def save_dataset():
    t = load_data()
    x_train, x_test, vocab = create_dataset(t)
    embeddings = create_embeddings(vocab)
    print(embeddings.shape)

    d = {}
    d['x_train'] = x_train
    d['x_test'] = x_test
    d['vocab'] = vocab
    d['pre_trained_embeddings'] = embeddings
    with open('data/prepared_dataset_for_pretrained_emb.p', 'wb') as f:
        pickle.dump(d, f)

save_dataset()