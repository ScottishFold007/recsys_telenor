from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

def create_target(sequence):
    target = sequence[1:]
    return np.append(target,[0])

def load_imdb_dataset():
    # load the dataset but only keep the top n words, zero the rest
    input_set_length = 200
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=input_set_length)
    target_set_length = input_set_length

    # truncate and pad input sequences
    #max_review_length = 500
    #X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    #X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    train = []
    test = []

    for x in X_train:
        train.append((x,create_target(x)))

    for x in X_test:
        test.append((x,create_target(x)))

    return train, test, input_set_length, target_set_length

train, test, input_set_length, target_set_length = load_imdb_dataset()
