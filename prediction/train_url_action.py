import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import math
import random

from url_action_model import Model
import url_action_preparation as uap

def accuracy(output):
    correct = 0
    for pred, target in outputs:
        if (pred == target):
            correct += 1
    return correct, correct/len(outputs)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
    
def train(model, train_data, loss_function, optimizer):
    outputs = []
    total_loss = 0.0
    for url_action_session, target_session in train_data:
        model.zero_grad()
        loss = 0.0
        hidden = None
        for url_action, target in zip(url_action_session, target_session):
            #print()
            output, hidden = model(url_action, hidden)
            l = loss_function(output, target.unsqueeze(0))
            loss += l
        #outputs.append((output.argmax().item(),target.item()))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return outputs, total_loss


train_data, test_data, url_action_set_length, url_set_length = uap.create_dataset()

print('url action set length',url_action_set_length)
print('url set length',url_set_length)
model = Model(url_action_set_length, url_set_length)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


n_iters = 100
print_every = 1
all_losses = []
test_every = 100
all_accuracies = []
start = time.time()

for iter in range(1, n_iters + 1):
    random.shuffle(train_data)
    outputs, loss = train(model, train_data, loss_function, optimizer)
    all_losses.append(loss)

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
    
    #correct, acc = accuracy(outputs)
    #all_accuracies.append(acc)
    #if iter % test_every == 0:
        #print('correct: %d accuracy: %.3f' % (correct, acc))
'''
test_out = []
for url, action, one_hot_last_url in test_data:
    output = model(url, action)
    outputs.append((output.argmax().item(),one_hot_last_url.item()))

c, a = accuracy(test_out)
print('correct: %d accuracy: %.3f' % (c, a))
'''
plt.figure()
plt.plot(all_losses)
plt.show()
'''
plt.figure()
plt.plot(all_accuracies)
plt.show()
'''