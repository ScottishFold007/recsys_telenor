import pandas as pd 

def prepare_urls(df):
    df['url'] = df['url'].apply(lambda x: x.rsplit('?', 1)[0])

    # split on last / and take first part of the url if url contains 'subscriptions' or 'tickets' or 'admins' or 'recommendations'
    df['url'] = df['url'].apply(lambda x: x.rsplit('/', 1)[0] if 'subscriptions' or 'tickets' or 'admins' or 'recommendations' in x else x) 

    # split on last / and take first part of the url if the url is not an empty string and the last element is digit
    df['url'] = df['url'].apply(lambda x: x.rsplit('/', 1)[0] if x and x[-1].isdigit() else x) 

    api_keywords = pd.read_csv('../preprocessing/API_keywords.txt')['keywords']

    for k in api_keywords:
        df['url'] = df['url'].apply(lambda x: k if x.find(k) != -1 else x)
    
    return df
