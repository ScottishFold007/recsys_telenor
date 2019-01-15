import pickle
import numpy as np
import uuid 
from datetime import datetime as dt
import pandas as pd

def define_session(df):

    pd.options.mode.chained_assignment = None


    # define time variables
    # define time variables
    df['start_time'] = df['start_time'].apply(lambda x: dt.strptime(str(x), "%Y-%m-%d %H:%M:%S:%f"))
    df['start_time'] = df['start_time'].apply(lambda x: x.replace(microsecond=0))

    df['date'] = df['start_time'].dt.date
    df['hour'] = df['start_time'].dt.hour
    df['DOW'] = df['start_time'].dt.dayofweek

    # create new session_id based on load = "new browser session"
    # visit_id is not a good measure, since people remain logged in for 1 hour. This was previously 2 hours.
    # in the App, people remain logged in for 11 months, so visit_ids could carry on for a long time
    # Advice: I would define a session based on inactivity. Create new session after 30 minutes inactivity
    df.sort_values(['visit_id', 'start_time'], inplace=True)

    df['lag_ts'] = df.sort_values(['visit_id','start_time']).groupby('visit_id')['start_time'].shift(1)
    df['lag_ts'].fillna(df['start_time'],inplace=True) # for the first event in session
    df['inactivity'] = (df['start_time'] - df['lag_ts']) / np.timedelta64(1, 'm')

    cond_inactivity = df.inactivity > 30
    cond_url_not_NaN = df.url is not np.nan
    cond_lag_ts_NaN = df.lag_ts is np.nan
    cond_login = ((df.url == 'https://www.telenor.no/bedrift/minbedrift/beta/#/') | (df.url == 'https://www.telenor.no/bedrift/minbedrift/beta/') | (df.url == 'https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/')) & ("_load_" in df.action)
    cond = cond_url_not_NaN & ((cond_login & cond_lag_ts_NaN) | cond_inactivity)
    

    df['tmp'] = cond.groupby(df.visit_id).cumsum().where(cond, 0).astype(int).replace(to_replace=0, method='ffill')

    df['sequence'] = df.groupby(['tmp', 'visit_id']).cumcount() + 1
    df['UUID'] = 1
    df.loc[:, "UUID"] = df.groupby(['user', 'tmp', 'visit_id'])['UUID'].transform(lambda g: uuid.uuid4())

    # drop all sessions with 1 event (since they are duplicates)
    df['uuid_count'] = df.groupby('UUID').UUID.transform('count')
    df = df[df.uuid_count > 1]
    print(len(df.index))
    return df

