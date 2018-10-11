import pickle
import numpy as np
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate

#dataset = pd.read_csv("./../../../splunk_data_180918_telenor.txt")

# take last 20 000 items
#df = dataset.tail(50)

df = pickle.load( open( "data_set.p", "rb" ) )

actions = df['action'].fillna('')
urls = df['url'].fillna('')

for u in urls:
    print(u)

print(actions)


def encode_documents(data):

    # create the tokenizer
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(data)

    vocabulary_size = len(t.word_index) + 1
    
    # integer encode the documents 
    encoded_documents = t.texts_to_sequences(data)
    
    # pad documents to a max length 
    max_length = len(max(encoded_documents, key=len))
    padded_documents = pad_sequences(encoded_documents, maxlen=17, padding='post')
    
    # create one-hot for each sequence of indices
    onehots = []
    for d in padded_documents:
        onehots.append(to_categorical(d,num_classes=vocabulary_size))

    return vocabulary_size, max_length, np.array(onehots)

url_vocabulary_size, url_max_length, encoded_urls = encode_documents(urls)
action_vocabulary_size, action_max_length, encoded_actions = encode_documents(actions)

print('url max length:', url_max_length)
print('action max length:', action_max_length)

print('urls vocabulary size:', url_vocabulary_size)
print('actions vocabulary size:', action_vocabulary_size)

print('urls shape:', encoded_urls.shape)
print('actions shape:', encoded_actions.shape)

# TODO: create dense matrix from url and action

# TODO: concat embeddings for url and action

# split data into training and testing
train_size = int(0.8 * len(encoded_urls))
train_urls = encoded_urls[0:train_size]
test_urls = encoded_urls[train_size:-1]
train_actions = encoded_actions[0:train_size - 1] # to handle look back
	
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
train_x_urls, train_y_urls = create_dataset(train_urls, look_back)
test_X_urls, test_y_urls = create_dataset(test_urls, look_back)

'''

# define the model
urls = Input(shape=train_x_urls.shape)
actions = Input(shape=train_actions.shape)
url_embedding = Embedding(input_dim=url_vocabulary_size, output_dim=8, input_length=17)(urls)
action_embedding = Embedding(input_dim=action_vocabulary_size, output_dim=8, input_length=17)(actions)
# flatten so that embeddings have same shape and can be concatenated
#flatten_urls = Flatten()(url_embedding)
#flatten_actions = Flatten()(action_embedding)
merge = concatenate(axis=-1, inputs=[url_embedding, action_embedding])
dense = Dense(1)(merge)
lstm = LSTM(units=1)(dense)
url_output = Dense(1)(lstm)
action_output = Dense(1)(lstm)
model = Model(inputs=[urls, actions], outputs=[url_output, action_output])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
#model.fit([train_urls, train_actions], test_urls, epochs=50, verbose=0)
# evaluate the model
#loss, accuracy = model.evaluate([train_urls, train_actions], test_urls, verbose=0)
#print('Accuracy: %f' % (accuracy*100))

# TODO: test LSTM on test data

# TODO: split preprocessing data and lstm into different scripts - save and load preprocessed data from file (pickle)

'''