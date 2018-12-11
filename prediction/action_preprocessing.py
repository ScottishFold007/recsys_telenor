import pickle
import numpy as np
import pandas as pd 
from datetime import datetime as dt
import torch
import uuid 
import spacy
#import en_core_web_sm
#import xx_ent_wiki_sm
#from spacy.lang.xx import MultiLanguage
#nlp = MultiLanguage()
#from spacy.lang.nb import Norwegian
#nlp = Norwegian()

t = pickle.load( open( "data_set.p", "rb" ) )
'''
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

t['lag_ts'] = t.sort_values(['visit_id','start_time']).groupby('visit_id')['start_time'].shift(1)
t['lag_ts'].fillna(t['start_time'],inplace=True) # for the first event in session
t['inactivity'] = (t['start_time'] - t['lag_ts']) / np.timedelta64(1, 'm')

cond_inactivity = t.inactivity > 20
cond_load = t.action == '_load_'
cond_homepage = t.url == 'https://www.telenor.no/bedrift/minbedrift/beta/#/'
cond = (cond_load & cond_homepage) | cond_inactivity
t['tmp'] = cond.groupby(t.visit_id).cumsum().where(cond, 0).astype(int).replace(to_replace=0, method='ffill')

t['sequence'] = t.groupby(['tmp', 'visit_id']).cumcount() + 1
t['UUID'] = 1
t.loc[:, "UUID"] = t.groupby(['user', 'tmp', 'visit_id'])['UUID'].transform(lambda g: uuid.uuid4())

# drop all sessions with 1 event (since they are duplicates)
t['uuid_count'] = t.groupby('UUID').UUID.transform('count')
t = t[t.uuid_count > 1]

# replace NaN with empty string
#t.fillna('',inplace=True)

# removing consecutive duplicate actions
t['prev_action'] = t.sort_values(['visit_id','start_time']).groupby('visit_id')['action'].shift(1)
t = t[t.action != t.prev_action]
print(t)
#print(len(set(t['UUID'])))
'''
'''
t['url'] = t['url'].apply(lambda x: x.rsplit('?', 1)[0])

# split on last / and take first part of the url if url contains 'subscriptions' or 'tickets' or 'admins' or 'recommendations'
t['url'] = t['url'].apply(lambda x: x.rsplit('/', 1)[0] if 'subscriptions' or 'tickets' or 'admins' or 'recommendations' in x else x) 

# split on last / and take first part of the url if the url is not an empty string and the last element is digit
t['url'] = t['url'].apply(lambda x: x.rsplit('/', 1)[0] if x and x[-1].isdigit() else x) 

api_keywords = pd.read_csv('./API_keywords.txt')['keywords']

for k in api_keywords:
    t['url'] = t['url'].apply(lambda x: k if x.find(k) != -1 else x)


url_set = set(t['url'])
action_set = set(t['action'])


for a in action_set:
    print(a)
'''

t.fillna('',inplace=True)
nb = spacy.load("nb_dep_ud_sm")


def tag(action):
    doc = nb(action)
    return [(X.text, X.label_) for X in doc.ents]


#a = 'click on "ULLENSVANG HERAD"'
a = "Det er kaldt på vinteren i Norge."
#nlp = en_core_web_sm.load() 
doc = nb(a)
print(doc)
print(doc.ents)
print([(X.text, X.label_) for X in doc.ents])

'''
#nlp = en_core_web_sm.load() 
t['action entities'] = t['action'].apply(tag)
t['curr'] = t['action']
print(t)
'''