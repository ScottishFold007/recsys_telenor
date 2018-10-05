import pandas as pd 
import string
import torch
from torch.utils.data import random_split
from keras.preprocessing.text import text_to_word_sequence
import itertools

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")
# print(dataset.values.shape)
#df = dataset.head(50)
df = dataset
# create events (action,url) using concatenation
events = (df["action"] + df["url"]).fillna('')
df = df.assign(event=events)
# print(df)
#print(events.values)
words = [text_to_word_sequence(e) for e in events.values]
seq = set(itertools.chain.from_iterable(words))
print(len(seq))
