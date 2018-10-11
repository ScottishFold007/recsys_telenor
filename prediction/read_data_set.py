import pickle
import pandas as pd 

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take last 20 000 items
df = dataset.tail(200)
#df = dataset

pickle.dump( df, open( "data_set.p", "wb" ) )