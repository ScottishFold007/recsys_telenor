import pickle
from datetime import datetime as dt
import torch
import numpy as np

import sys
sys.path.insert(0, './../preprocessing')
import session as ss
import url_preparation as up

def load_data():
    t = pickle.load( open( "./../data_set.p", "rb" ) )
    t = t.tail(300000)
    print(len(t.index))
    t = ss.define_session(t)
    return up.prepare_urls(t)

def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def create_target(sequence):
    target = sequence[1:]
    return np.append(target,[0]) 

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    train = dataset[0:train_size]
    test = dataset[train_size:-1]
    return train, test

def create_dataset(df, input_column, target_column):
    target_set = list(set(df[target_column]))
    target_set.insert(0,'ZERO PADDING')
    target_set_length = len(target_set)
    
    input_set = list(set(df[input_column]))
    input_set.insert(0,'ZERO PADDING')
    input_set_length = len(input_set)
    
    input_indices = integer_encode(input_set, df[input_column])
    target_indices = integer_encode(target_set,df[target_column])
    df = df.assign(target_index=target_indices)
    df = df.assign(input_index=input_indices)

    x = [] 
    y = []
    for uuid, row in df.groupby('UUID'):
        x.append(torch.LongTensor(row['input_index'].values))
        y.append(torch.LongTensor(create_target(row['target_index'].values)))

    # split data into training and testing
    x_train, x_test = split_dataset(x)
    y_train, y_test = split_dataset(y)

    return x_train, y_train, x_test, y_test, input_set_length, target_set_length, input_set

def create_dataset_url_action():
    t = load_data()

    # concat url and action
    t['url_action'] = t[['url_cleaned', 'action_cleaned']].apply(lambda x: ' '.join(x), axis=1)
    return create_dataset(t, 'url_action', 'url_cleaned')

def create_dataset_url():
    t = load_data()
    return create_dataset(t,'url_cleaned','url_cleaned')

def create_dataset_action():
    t = load_data()
    return create_dataset(t,'action_cleaned','action_cleaned')


