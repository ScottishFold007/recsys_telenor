import pandas as pd 

def prepare_urls(df):
    api_keywords = pd.read_csv('../preprocessing/API_keywords.txt')['keywords']

    for k in api_keywords:
        df['url_cleaned'] = df['url'].apply(lambda x: k if k in x else x)
    
    return df
