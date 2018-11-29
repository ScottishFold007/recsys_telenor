import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        #self.hidden_size = hidden_size

        self.embedding_1 = nn.Embedding(num_embeddings=100, embedding_dim=url_set_length)
        self.embedding_2 = nn.Embedding(num_embeddings=100, embedding_dim=action_set_length)
        self.linear = nn.Linear(in_features=106, out_features=106)
        self.lstm = nn.LSTM(input_size=106, hidden_size=106, num_layers=2, dropout=0.05)
        self.linear_out = nn.Linear(in_features=106, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, x1, x2):
        embedding_1 = self.embedding_1(x1)
        embedding_2 = self.embedding_2(x2)
        concat = torch.cat((embedding_1.data, embedding_2.data))
        concat = concat.unsqueeze(0)
        concat = concat.unsqueeze(0)
        linear = self.linear(concat)
        lstm, hidden = self.lstm(linear)
        output = self.linear_out(lstm)
        return output 

   
def train(lstm, input_1, input_2, target, criteron, optimizer):
    lstm.zero_grad()
    loss = 0.0
    outputs = []
    for i in range(input_1.shape[0]):
        #print(input_1[i])
        #print(input_2[i])
       # print(target[i])
        output = lstm(input_1[i],input_2[i])
        l = criterion(output[0][0], target[0][0][i])
        loss += l
        outputs.append(output.data.numpy()[0][0][0])
    loss.backward()
    optimizer.step()
    return outputs, loss.item()

def integer_encode(vocabulary, data):
    # mapping from urls to int
    string_to_int = dict((s,i) for i,s in enumerate(vocabulary))

    # integer encode
    integer_encoded = [string_to_int[s] for s in data]
    return integer_encoded

def one_hot_encoding(vocabulary, data):
    integer_encoded = integer_encode(vocabulary, data)
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


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


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

'''
one_hot_urls = one_hot_encoding(url_set, urls)
one_hot_actions = one_hot_encoding(action_set, actions)

one_hot_urls = np.array(one_hot_urls)
one_hot_actions = np.array(one_hot_actions)
'''
integer_urls = integer_encode(url_set,urls)
integer_actions = integer_encode(action_set,actions)

integer_urls = np.array(integer_urls)
integer_actions = np.array(integer_actions)

# split data into training and testing
train_size = int(0.8 * len(integer_urls))
train_urls = integer_urls[0:train_size]
test_urls = integer_urls[train_size:-1]
train_actions = integer_actions[0:train_size - 1] # to handle look back
test_actions = integer_actions[train_size:-2]

# reshape into X=t and Y=t+1
look_back = 1
train_x_urls, train_y_urls = create_dataset(train_urls, look_back)
test_x_urls, test_y_urls = create_dataset(test_urls, look_back)

lstm = LSTM()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(lstm.parameters())

n_iters = 20
print_every = 1
test_every = 10
all_losses = []

start = time.time()

u = torch.from_numpy(train_x_urls).long()
a = torch.from_numpy(train_actions).long()
target = torch.from_numpy(train_y_urls).unsqueeze(0).float()
target = target.unsqueeze(0)
print(urls.shape,actions.shape,target.shape)
#print(actions)
for iter in range(1, n_iters + 1):
    outputs, loss = train(lstm, u, a, target, criterion, optimizer)
    all_losses.append(loss)

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
    
    if iter % test_every == 0:
        rounded_outputs = np.rint(outputs)
        correct = 0
        for pred, label in zip(rounded_outputs,train_y_urls):
            if (pred == label):
                correct += 1
        print('correct: %d %.3f' % (correct, correct/len(train_y_urls)))

u_t = torch.from_numpy(test_x_urls).long()
a_t = torch.from_numpy(test_actions).long()

correct = 0
for i in range(u_t.shape[0]):
    output = lstm(u_t[i],a_t[i])
    pred = np.rint(output.data.numpy()[0][0][0])
    if (pred == test_y_urls[i]):
        correct += 1
print('test correct %d %.3f' % (correct, correct/len(test_y_urls)))

plt.figure()
plt.plot(all_losses)
plt.show()
