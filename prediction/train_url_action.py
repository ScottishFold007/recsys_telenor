import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import math
from tensorboardX import SummaryWriter

from url_action_model import Model
import url_action_preparation as uap 
import train_dataset 
import test_dataset

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def test_accuracy(model, x, y, session_size, batch_size, url_set_length, device):
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, None)
    output = output.view(batch_size, url_set_length, session_size)

    # accuracy
    prediction = torch.argmax(output, dim=1)
    num_correct = torch.sum(prediction == targets)
    accuracy = num_correct.item()/session_size
    del inputs
    del targets
    return accuracy/batch_size

def train(model, x, y, loss_function, optimizer, session_size, batch_size, url_set_length, decive):
    model.zero_grad()
    loss = 0.0
    hidden = None
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, hidden)
    output = output.view(batch_size, url_set_length, session_size) # according to pytorch CE api input format: (minibatch size, #classes, d)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()

    # accuracy
    prediction = torch.argmax(output, dim=1)
    num_correct = torch.sum(prediction == targets)
    accuracy = num_correct.item()/session_size

    del inputs 
    del targets
    del output
    return loss.item(), accuracy/batch_size

writer = SummaryWriter('logs') 
train_data, test_data, url_action_set_length, url_set_length = uap.create_dataset()
training_dataset = train_dataset.TrainDataset(train_data)
testing_dataset = test_dataset.TestDataset(test_data)


print('url action set length',url_action_set_length)
print('url set length',url_set_length)
print('training size',len(train_data))
print('test size',len(test_data))
session_size = len(train_data[0][0])
print('session size',session_size)

n_iters = 300
print_every = 5
test_every = 10
train_loss = []
train_accuracies = []
test_accuracies = []
batch_size = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(url_action_set_length, 10, url_set_length, batch_size, session_size, device)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
train_count = 0
test_count = 0
start = time.time()
for i in range(1, n_iters + 1):
    epoch_loss = []
    for j, minibatch in enumerate(train_loader, 0):
        x, y = minibatch
        loss, accuracy = train(model, x, y, loss_function, optimizer, session_size, batch_size, url_set_length, device)
        writer.add_scalar('Train/Loss', loss, train_count)
        writer.add_scalar('Train/Accuracy', accuracy, train_count)
        train_accuracies.append(accuracy)
        epoch_loss.append(loss)
        train_count += 1
    train_loss.append(epoch_loss)

    if i % print_every == 0:
        print('%s (%d %d%%) %.4f %.4f' % (timeSince(start), i, i / n_iters * 100, sum(epoch_loss), accuracy))
    
    if i % test_every == 0:
        for j, minibatch in enumerate(test_loader, 0): 
            x, y = minibatch
            test_acc = test_accuracy(model, x, y, session_size, batch_size, url_set_length, device)
            test_accuracies.append(test_acc)
            writer.add_scalar('Test/Accuracy', test_acc, train_count)
            test_count += 1
    

plt.figure()
plt.plot(test_accuracies)
plt.savefig('img/test_acc.png')
plt.plot(train_accuracies)
plt.savefig('img/train_acc.png')
plt.plot(train_loss)
plt.savefig('img/loss.png')


  