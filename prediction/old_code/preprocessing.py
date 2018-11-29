import pickle
import numpy as np
import pandas as pd 

df = pickle.load( open( "data_set.p", "rb" ) )

actions = df['action'].fillna('')
urls = df['url'].fillna('')

api_keywords = pd.read_csv('./API_keywords.txt')['keywords']

for k in api_keywords:
    urls = urls.apply(lambda x: k if x.find(k) != -1 else x)

url_set = set(urls)
action_set = set(actions)

'''
for u in url_set:
    print(u)
'''

for a in action_set:
    print(a)

url_set_length = len(url_set)
action_set_length = len(action_set)

print('url set length', url_set_length)
print('actions set length', action_set_length)


