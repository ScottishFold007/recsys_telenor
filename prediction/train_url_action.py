import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import math
import numpy as np
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

def test_accuracy(model, x, y, batch_size, url_set_length, device):
    session_size = x.shape[1]
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, None)
    output = output.view(batch_size, url_set_length, session_size)
    accuracy = calc_accuracy(output, targets, session_size, batch_size)
    del inputs
    del targets
    return accuracy

def calc_accuracy(output_distribution, targets, session_size, batch_size):
    prediction = torch.argmax(output_distribution, dim=1)
    mask = targets != 0
    correct_with_padding = prediction == targets
    correct_class = mask * correct_with_padding
    num_correct = torch.sum(correct_class)
    return (num_correct.item()/session_size)/batch_size

def train(model, x, y, loss_function, optimizer, batch_size, url_set_length, device):
    session_size = x.shape[1]
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, None)
    output = output.view(batch_size, url_set_length, session_size) # according to pytorch CE api input format: (minibatch size, #classes, d)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()

    train_accuracy = calc_accuracy(output, targets, session_size, batch_size)
    del inputs 
    del targets
    del output
    return loss.item(), train_accuracy

def pad_minibatch(minibatch):
    max_len = max(len(s[0]) for s in minibatch)
    sequences = [] 
    targets = []
    #print(minibatch[1])
    for seq, target in minibatch:
        pad = np.full((max_len - len(seq)),0)
        seq = np.concatenate((seq,pad)) 
        target = np.concatenate((target,pad)) 
        sequences.append(seq)
        targets.append(target)
    return torch.LongTensor(sequences), torch.LongTensor(targets) 
    

writer = SummaryWriter('logs') 
train_data, test_data, url_action_set_length, url_set_length = uap.create_dataset()
training_dataset = train_dataset.TrainDataset(train_data)
testing_dataset = test_dataset.TestDataset(test_data)


print('url action set length',url_action_set_length)
print('url set length',url_set_length)
print('training size',len(train_data))
print('test size',len(test_data))

n_iters = 1
print_every = 5
test_every = 1
train_loss = []
train_accuracies = []
test_accuracies = []
batch_size = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model(
    vocabulary_size=url_action_set_length,
    embedding_size=10,
    output_size=url_set_length, 
    minibatch_size=batch_size, 
    device=device)

model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, collate_fn=pad_minibatch, shuffle=False, num_workers=4, drop_last=True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, collate_fn=pad_minibatch, shuffle=False, num_workers=4, drop_last=True)
train_count = 0
test_count = 0
start = time.time()
for i in range(1, n_iters + 1):
    epoch_loss = []
    for j, minibatch in enumerate(train_loader, 0):
        x, y = minibatch
        loss, accuracy = train(
            model=model, 
            x=x, 
            y=y, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            batch_size=batch_size, 
            url_set_length=url_set_length, 
            device=device)

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
            test_acc = test_accuracy(
                model=model, 
                x=x, 
                y=y, 
                batch_size=batch_size, 
                url_set_length=url_set_length, 
                device=device)
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


  