import pickle
import pandas as pd 

in_path = './../../'
t = pd.read_csv(
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


t.columns = t.columns.str.replace('min_bedrift_event.','')
t = t[~t.action.isnull()]

# drop NaN actions or urls
t = t.dropna(axis='rows', how='any',subset=['url', 'action'])
t = t.reset_index()

pickle.dump( t, open( "data_set.p", "wb" ) )