import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import math
import random
from tensorboardX import SummaryWriter

from url_action_model import Model
import url_action_preparation as uap  

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def test_accuracy(model, test_data, session_size):
    test_size = len(test_data)
    test_accuracy = []
    for url_action_session, target_session in test_data:
        hidden = None
        x = url_action_session.to(device)
        t = target_session.to(device)
        output, hidden = model(x, hidden)

        # accuracy
        prediction = torch.argmax(output, dim=1)
        num_correct = torch.sum(prediction == t)
        test_accuracy.append(num_correct.item())
        del x 
        del t
    return (sum(test_accuracy)/session_size)/test_size

def train(model, train_data, loss_function, optimizer, session_size, batch_size, decive):
    outputs = []
    total_loss = 0.0
    accuracy = 0.0
    for url_action_session, target_session in train_data:
        model.zero_grad()
        loss = 0.0
        hidden = None
        x = url_action_session.to(device)
        t = target_session.to(device)
        output, hidden = model(x, hidden)
        loss = loss_function(output, t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # accuracy
        prediction = torch.argmax(output, dim=1)
        num_correct = torch.sum(prediction == t)
        accuracy += num_correct.item()/session_size
        del x
        del t
    return outputs, total_loss, accuracy/batch_size

writer = SummaryWriter('logs') 
train_data, test_data, url_action_set_length, url_set_length = uap.create_dataset()

print('url action set length',url_action_set_length)
print('url set length',url_set_length)
print('training size',len(train_data))
print('test size',len(test_data))
session_size = len(train_data[0][0])
print('session size',session_size)

n_iters = 1000
print_every = 5
test_every = 100
all_losses = []
train_accuracies = []
test_accuracies = []
batch_size = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(url_action_set_length, 10, url_set_length, device)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

start = time.time()
for i in range(1, n_iters + 1):
    #random.shuffle(train_data)
    minibatch = random.sample(train_data, batch_size)
    
    outputs, loss, accuracy = train(model, minibatch, loss_function, optimizer, session_size, batch_size, device)
    writer.add_scalar('Train/Loss', loss, i)
    writer.add_scalar('Train/Accuracy', accuracy, i)
    train_accuracies.append(loss)

    if i % print_every == 0:
        print('%s (%d %d%%) %.4f %.4f' % (timeSince(start), i, i / n_iters * 100, loss, accuracy))

    if i % test_every == 0:
        test_acc = test_accuracy(model, test_data, session_size)
        test_accuracies.append(test_acc)
        writer.add_scalar('Test/Accuracy', test_acc, i)


plt.figure()
plt.plot(test_accuracies)
plt.savefig('img/test_acc.png')
plt.plot(train_accuracies)
plt.savefig('img/train_acc.png')
plt.plot(all_losses)
plt.savefig('img/loss.png')