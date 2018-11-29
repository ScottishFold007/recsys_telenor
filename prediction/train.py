import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import math

from model import Model
import datapreparation as datp


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
    
def train(model, train_data, loss_function, optimizer):
    model.zero_grad()
    loss = 0.0
    outputs = []
    for url, action, target in train_data:
        print(target.shape)
        output = model(url, action)
        l = loss_function(output, target)
        loss += l
        #outputs.append((output.argmax().item(),target.item()))
    loss.backward()
    optimizer.step()
    return outputs, loss.item()

def accuracy(output):
    correct = 0
    for pred, target in outputs:
        if (pred == target):
            correct += 1
    return correct, correct/len(outputs)

train_data, test_data, url_set_length, action_set_length = datp.create_dataset()

model = Model(url_set_length, action_set_length)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

n_iters = 1000
print_every = 10
all_losses = []
test_every = 100
all_accuracies = []
start = time.time()

for iter in range(1, n_iters + 1):
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