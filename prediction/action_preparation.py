import pickle
from datetime import datetime as dt
import torch
import numpy as np
import random

import sys
sys.path.insert(0, './../preprocessing')


def load_data():
    '''
    returns a pandas dataframe where each grouped by session id (UUID) and time of origin. 
    Each row is a tupple of containing interaction and additional context data.
    '''
    t = pickle.load( open( "./data/cleaned_dataset.p", "rb" ) )
    print(len(t.index))
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

def create_dataset(df, input_column):
    
    vocab = list(set(df[input_column]))
    
    input_indices = integer_encode(vocab, df[input_column])
    df = df.assign(input_index=input_indices)

    x = [] 
    for uuid, row in df.groupby('UUID'):
        x.append(torch.LongTensor(row['input_index'].values))

    # split data into training and testing
    x_train, x_test = split_dataset(x)

    return x_train, x_test, vocab

def create_dataset_action():
    t = load_data()
    return create_dataset(t,'action_cleaned')

def save_dataset():
    x_train, x_test, vocab = create_dataset_action()
    print(len(x_train))
    d = {}
    d['x_train'] = x_train
    d['x_test'] = x_test
    d['vocab'] = vocab
    with open('prepared_dataset.p', 'wb') as f:
        pickle.dump(d, f)

save_dataset()




