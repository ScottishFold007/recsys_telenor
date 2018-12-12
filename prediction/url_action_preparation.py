import pickle
import numpy as np
import pandas as pd 
from datetime import datetime as dt
import torch

def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def one_hot_encode(vocabulary, integer_encoded):
    onehot_encoded = list()
    for i in integer_encoded:
        word = [0 for _ in range(len(vocabulary))]
        if i != 0:
            word[i-1] = 1
        onehot_encoded.append(word)
    return np.array(onehot_encoded)

def create_target(sequence):
    target = sequence[1:]
    return np.append(target,[0])


def create_dataset():
    import uuid # TODO: remove ugly hack 

    t = pickle.load( open( "data_set.p", "rb" ) )

     # drop rows with NaN 
    t = t.dropna(axis='rows', how='any',subset=['url', 'action'])

    # define time variables
    t['start_time'] = t['start_time'].apply(lambda x: dt.strptime(str(x), "%Y-%m-%d %H:%M:%S:%f"))
    t['start_time'] = t['start_time'].apply(lambda x: x.replace(microsecond=0))

    t['date'] = t['start_time'].dt.date
    t['hour'] = t['start_time'].dt.hour
    t['DOW'] = t['start_time'].dt.dayofweek

    # create new session_id based on load = "new browser session"
    # visit_id is not a good measure, since people remain logged in for 1 hour. This was previously 2 hours.
    # in the App, people remain logged in for 11 months, so visit_ids could carry on for a long time
    # Advice: I would define a session based on inactivity. Create new session after 30 minutes inactivity
    t.sort_values(['visit_id', 'start_time'], inplace=True)
    cond = t.action == '_load_'
    t['tmp'] = cond.groupby(t.visit_id).cumsum().where(cond, 0).astype(int).replace(to_replace=0, method='ffill')

    t['sequence'] = t.groupby(['tmp', 'visit_id']).cumcount() + 1
    t['UUID'] = 1
    t.loc[:, "UUID"] = t.groupby(['user', 'tmp', 'visit_id'])['UUID'].transform(lambda g: uuid.uuid4())

    # drop all sessions with 1 event (since they are duplicates)
    t['uuid_count'] = t.groupby('UUID').UUID.transform('count')
    t = t[t.uuid_count > 1]

    t['url'] = t['url'].apply(lambda x: x.rsplit('?', 1)[0])

    # split on last / and take first part of the url if url contains 'subscriptions' or 'tickets' or 'admins' or 'recommendations'
    t['url'] = t['url'].apply(lambda x: x.rsplit('/', 1)[0] if 'subscriptions' or 'tickets' or 'admins' or 'recommendations' in x else x) 

    # split on last / and take first part of the url if the url is not an empty string and the last element is digit
    t['url'] = t['url'].apply(lambda x: x.rsplit('/', 1)[0] if x and x[-1].isdigit() else x) 

    api_keywords = pd.read_csv('./API_keywords.txt')['keywords']

    for k in api_keywords:
        t['url'] = t['url'].apply(lambda x: k if x.find(k) != -1 else x)

    
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