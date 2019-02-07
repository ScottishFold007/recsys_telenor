import pandas as pd
import numpy as np 
import torch

def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))
    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    train = dataset[0:train_size]
    test = dataset[train_size:-1]
    return train, test

def create_target(sequence):
    target = sequence[1:]
    return np.append(target,[0]) 

def load_data():
    df = pd.read_csv('./../../events.csv').head(20000)

    input_set = list(set(df['itemid']))
    input_set.insert(0,'ZERO PADDING')
    input_set_length = len(input_set)
    target_set_length = input_set_length

    input_indices = integer_encode(input_set, df['itemid'])
    df = df.assign(input_index=input_indices)

    x = []
    y = []
    for visit_id, row in df.groupby(df.visitorid):
        vals = row['input_index'].values
        if len(vals) > 1:
            x.append(torch.LongTensor(vals))
            y.append(torch.LongTensor(create_target(vals)))

    # split data into training and testing
    x_train, x_test = split_dataset(x)
    y_train, y_test = split_dataset(y)

    return x_train, y_train, x_test, y_test, input_set_length, target_set_length





