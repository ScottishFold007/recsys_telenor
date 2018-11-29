import pickle
import numpy as np
import pandas as pd 
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate


def one_hot_encoding(vocabulary, data):

    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    # print(integer_encoded)

    # one-hot encode
    onehot_encoded = list()
    for i in integer_encoded:
        word = [0 for _ in range(len(vocabulary))]
        word[i] = 1
        onehot_encoded.append(word)

    # print(onehot_encoded)
    return onehot_encoded

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

df = pickle.load( open( "data_set.p", "rb" ) )

actions = df['action'].fillna('')
urls = df['url'].fillna('')

# split on last ? and take first part of the url
urls = urls.apply(lambda x: x.rsplit('?', 1)[0])

# split on last / and take first part of the url if url contains 'subscriptions' or 'tickets' or 'admins' or 'recommendations'
urls = urls.apply(lambda x: x.rsplit('/', 1)[0] if 'subscriptions' or 'tickets' or 'admins' or 'recommendations' in x else x) 

# split on last / and take first part of the url if the url is not an empty string and the last element is digit
urls = urls.apply(lambda x: x.rsplit('/', 1)[0] if x and x[-1].isdigit() else x) 

url_set = set(urls)
action_set = set(actions)

url_set_length = len(url_set)
action_set_length = len(action_set)

print('url set length', url_set_length)
print('actions set length', action_set_length)

# pad strings to url set to make it same length as action set
'''
padding = '-'
for x in range(action_set_length - url_set_length):
    url_set.add(padding)
    padding += '-'
'''
# print(url_set)

one_hot_urls = one_hot_encoding(url_set, urls)
one_hot_actions = one_hot_encoding(action_set, actions)

one_hot_urls = np.array(one_hot_urls, dtype='float32')
one_hot_actions = np.array(one_hot_actions,dtype='float32')

print('url array shape',one_hot_urls.shape)
print('action array shape',one_hot_actions.shape)

# split data into training and testing
train_size = int(0.8 * len(one_hot_urls))
train_urls = one_hot_urls[0:train_size]
test_urls = one_hot_urls[train_size:-1]
train_actions = one_hot_actions[0:train_size - 1] # to handle look back
test_actions = one_hot_actions[train_size:-2]

# reshape into X=t and Y=t+1
look_back = 1
train_x_urls, train_y_urls = create_dataset(train_urls, look_back)
test_x_urls, test_y_urls = create_dataset(test_urls, look_back)

# reshape input to be [samples, time steps, features]
# TODO: check if its really necessary
train_x_urls = train_x_urls[:, np.newaxis, :]
test_x_urls = test_x_urls[:, np.newaxis, :]
train_actions = train_actions[:, np.newaxis, :]
test_actions = test_actions[:, np.newaxis, :]

# define the model
urls = Input(shape=(train_x_urls.shape[1], train_x_urls.shape[2]))
actions = Input(shape=(train_actions.shape[1], train_actions.shape[2]))
url_embedding = Embedding(input_dim=url_set_length, output_dim=8, input_length=train_x_urls.shape[2])(urls)
action_embedding = Embedding(input_dim=action_set_length, output_dim=8, input_length=train_actions.shape[2])(actions)
merge = concatenate(axis=-1, inputs=[url_embedding, action_embedding])
dense = Dense(units=1, input_dim=train_x_urls.shape[2])(merge)
lstm = LSTM(units=1)(dense)
url_output = Dense(units=train_x_urls.shape[2], input_dim=train_x_urls.shape[2])(lstm)
model = Model(inputs=[urls, actions], outputs=url_output)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit([train_x_urls, train_actions], train_y_urls, epochs=50, batch_size=32, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate([test_x_urls, test_actions], test_y_urls, verbose=0)
print('Accuracy: %f' % (accuracy*100))




