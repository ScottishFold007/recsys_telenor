import pickle
from datetime import datetime as dt
import uuid 
import gensim
import numpy as np
import pandas as pd

def model_to_CSV(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tokens_df = pd.DataFrame(tokens)
    labels_df = pd.DataFrame(labels)
    tokens_df.to_csv('tokens.csv',sep='\t', index=False, header=None)
    labels_df.to_csv('labels.csv',sep='\t', index=False, header=None)

t = pickle.load( open( "data_set.p", "rb" ) )
t = t.dropna(axis='rows', how='any',subset=['url', 'action'])

 # define time variables
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

corpus = []
for uuid, row in t.groupby('UUID'):
    session = []
    for a in row['action']:
        session.append(a)
    corpus.append(session)

model = gensim.models.Word2Vec(corpus, size=150, window=5, min_count=2, workers=10)
model.train(corpus,total_examples=len(corpus),epochs=10)

model_to_CSV(model)




