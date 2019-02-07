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
import session as ss

t = pickle.load( open( "./../data_set.p", "rb" ) )

t = ss.define_session(t)

t = prep.clean_actions(t)


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
#t['ac'] = t['action_cleaned']
#t['a'] = t['action']
#t['action_cleaned'] = t['action'].apply(clean)
#t['action entities'] = t['action_cleaned'].apply(tag)

tt = t[t['action_cleaned']=='click_on_other'].action.unique()
clen = pd.DataFrame({'action':tt})
clen['action_cleaned'] = clen['action'].apply(clean) 
clen ['action entities'] = clen['action_cleaned'].apply(tag)
