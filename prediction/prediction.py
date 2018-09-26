import pandas as pd 
import torch
from torch.utils.data import random_split

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")
print(dataset.values.shape)
df = dataset.head(50)

# split data into training and testing

train_size = int(0.8 * len(df))
train_dataset = df[0:train_size]
test_dataset = df[train_size:-1]
print(train_dataset)
print(test_dataset)
