import pickle
import pandas as pd 

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take last 20 000 items
# was 200
t = dataset.tail(10000)

pickle.dump( t, open( "data_set.p", "wb" ) )