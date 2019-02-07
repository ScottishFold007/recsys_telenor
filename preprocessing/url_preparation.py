import pandas as pd 

def prepare_urls(df):
    api_keywords = pd.read_csv('../preprocessing/API_keywords.txt')['keywords']

    df['url_cleaned'] = df['url']
    for k in api_keywords:
        df['url_cleaned'] = df['url_cleaned'].apply(lambda x: k if k in x else x)   
    
    return df
