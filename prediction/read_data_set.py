import pickle
import pandas as pd 

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take last 20 000 items
t = dataset.tail(200)

pickle.dump( t, open( "data_set.p", "wb" ) )