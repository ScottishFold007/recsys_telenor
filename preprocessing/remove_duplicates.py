from datetime import datetime as dt
import uuid 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

import sys
sys.path.insert(0, './../preprocessing')
import session as ss

in_path = './../../../'
dataset = pd.read_csv(
    in_path+'splunk_data_180918_telenor_processed.txt',  
    encoding="ISO-8859-1", 
    dtype={
        "user_id": int, 
        "visit_id": int, 
        "sequence": int, 
        "start_time":object, 
        "event_duration":float,
        "url":str, 
        "action":str, 
        "country":str,
        "user_client":str,
        "user_client_family":str,
        "user_experience":str,
        "user_os":str,
        "apdex_user_experience":str,
        "bounce_rate":float,
        "session_duration":float
    }
)
t = dataset
t.columns = t.columns.str.replace('min_bedrift_event.','')
t = t[~t.action.isnull()]

# drop NaN actions or urls
t = t.dropna(axis='rows', how='any',subset=['url', 'action'])
print(len(t.index))

t = ss.define_session(t)

print('sessions defined')

def remove(s):
    s = s.loc[s.action_cleaned.shift() != s.action_cleaned]
    return s

t = t.groupby(['UUID','start_time']).apply(remove)

with open('cleaned_dataset.p', 'wb') as f:
    pickle.dump(t, f)