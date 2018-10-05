import pandas as pd 
import numpy as np
from numpy import newaxis
import math
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take first 50
df = dataset.tail(20000)

# create events (action,url) using concatenation
events = (df["action"] + df["url"]).fillna('')
df = df.assign(event=events)

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(events)

# summarize what was learned
'''
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
'''

encoded_docs = t.texts_to_matrix(events, mode='count')

# split data into training and testing
train_size = int(0.8 * len(encoded_docs))
train = encoded_docs[0:train_size]
test = encoded_docs[train_size:-1]
	
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = trainX[:, newaxis, :]
testX = testX[:, newaxis, :]

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=4, input_shape=(1,trainX.shape[2])))
model.add(Dense(trainX.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
