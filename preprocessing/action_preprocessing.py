import pickle
import numpy as np
import pandas as pd 
from datetime import datetime as dt
import torch
import uuid 
import spacy
import string
import en_core_web_sm
#import xx_ent_wiki_sm
#from spacy.lang.xx import MultiLanguage
#nlp = MultiLanguage()
#from spacy.lang.nb import Norwegian
#nlp = Norwegian() 

import preprocess as prep

t = prep.clean_actions()

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


#nb = spacy.load("nb_dep_ud_sm")
nlp = en_core_web_sm.load() 

import re 
regex = re.compile('[%s]' % re.escape(string.punctuation)) 

def clean(action):
    cleaned = regex.sub(' ', action)

    # remove multiple whitespace
    return re.sub(' +',' ',cleaned) 

def tag(action):
    doc = nlp(action)
    return [(X.text, X.label_) for X in doc.ents]

'''
#a = 'click on "ULLENSVANG HERAD"'
a = "Det er kaldt på vinteren i Norge."
#nlp = en_core_web_sm.load() 
doc = nb(a)
print(doc)
print(doc.ents)
print([(X.text, X.label_) for X in doc.ents])
'''

nlp = en_core_web_sm.load() 
t['ac'] = t['action_cleaned']
t['a'] = t['action']
#t['action_cleaned'] = t['action'].apply(clean)
#t['action entities'] = t['action_cleaned'].apply(tag)


#tt = t[t['action_cleaned']=='click_on_other'].action.unique()
print(t.action.value_counts().tail(50))