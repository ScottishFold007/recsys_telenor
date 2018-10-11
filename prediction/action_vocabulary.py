import pickle
import numpy as np
import pandas as pd 

df = pickle.load( open( "data_set.p", "rb" ) )

actions = df['action'].fillna('')
urls = df['url'].fillna('')

for a in actions:
    print(a)



print('------------------------------------------------------------------------------------- new -------------------------------------------------------------------------------------')


#x = urls.iloc[8]
#print(x)
#print(x.rsplit('?',1))
#print(x[-1].isdigit())
'''
# split on last ? and take first part of the url
urls = urls.apply(lambda x: x.rsplit('?', 1)[0])

# split on last / and take first part of the url if url contains 'subscriptions' or 'tickets' or 'admins' or 'recommendations'
urls = urls.apply(lambda x: x.rsplit('/', 1)[0] if 'subscriptions' or 'tickets' or 'admins' or 'recommendations' in x else x) 

# split on last / and take first part of the url if the url is not an empty string and the last element is digit
urls = urls.apply(lambda x: x.rsplit('/', 1)[0] if x and x[-1].isdigit() else x) 
distinct_urls = set(urls)


for u in distinct_urls:
    print(u)
'''