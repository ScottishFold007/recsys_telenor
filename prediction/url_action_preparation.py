import pickle
from datetime import datetime as dt
import torch

import sys
sys.path.insert(0, './../preprocessing')
import session as ss
import url_preparation as up

def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def create_target(sequence):
    target = sequence[1:]
    return np.append(target,[0])


def create_dataset():
    import uuid # TODO: remove ugly hack 

    t = pickle.load( open( "./../data_set.p", "rb" ) )

    t = ss.define_session(t)

    t = up.prepare_urls(t)
    
    url_set = list(set(t['url']))
    url_set.insert(0,'ZERO PADDING')
    action_set = set(t['action'])

    url_set_length = len(url_set)
    action_set_length = len(action_set)
    
    t['url_action'] = t[['url', 'action']].apply(lambda x: ' '.join(x), axis=1)
    url_action_set = list(set(t['url_action']))
    url_action_set.insert(0,'ZERO PADDING')
    url_action_set_length = len(url_action_set)
    
    url_action_indices = integer_encode(url_action_set, t['url_action'])
    url_indices = integer_encode(url_set, t['url'])
    t = t.assign(url_index=url_indices)
    t = t.assign(url_action_index=url_action_indices)

    dataset = []
    for uuid, row in t.groupby('UUID'):
        url_action = row['url_action_index'].values
        urls = row['url_index'].values

        target_urls = create_target(urls)

        #url_action_tensor = torch.from_numpy(url_action)
        #target_tensor = torch.from_numpy(target_urls)
        dataset.append((url_action,target_urls))

    # split data into training and testing
    train_size = int(0.8 * len(dataset))
    train = dataset[0:train_size]
    test = dataset[train_size:-1]

    train.sort(key = lambda s: len(s[0]))
    test.sort(key = lambda s: len(s[0]))

    return train, test, url_action_set_length, url_set_length

'''  
train, test, url_action_set_length, url_set_length = create_dataset()

print("train")
for t in train:
    print(t)
'''