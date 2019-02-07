import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import torch

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    train = dataset[0:train_size]
    test = dataset[train_size:-1]
    return train, test

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def toLongTensor(data):
    tensor_list = []
    for d in data:
        tensor_list.append(torch.LongTensor(d))
    return tensor_list

def prep_republic():
    # load
    in_filename = '../../republic_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output
    sequences = np.array(sequences)
    #sequences = sequences[0:50000]
    print(sequences.shape)
    X = sequences
    y = sequences[:,1:]
    y = np.hstack((y,np.zeros((y.shape[0],1))))

    # to tensor
    X = toLongTensor(X)
    y = toLongTensor(y)

    # split data into training and testing
    x_train, x_test = split_dataset(X)
    y_train, y_test = split_dataset(y)

    print('prep done')
    return x_train, y_train, x_test, y_test, vocab_size, vocab_size
