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

def test_accuracy(model, x, y, sequence_lengths, batch_size, url_set_length, device):
    session_size = x.shape[1]
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, sequence_lengths, None)
    output = output.view(batch_size, url_set_length, session_size)
    accuracy = calc_accuracy(output, targets)
    del inputs
    del targets
    return accuracy

def calc_accuracy(output_distribution, targets):
    prediction = torch.argmax(output_distribution, dim=1)
    mask = targets != 0
    correct_with_padding = prediction == targets
    correct_class = mask * correct_with_padding
    num_correct = torch.sum(correct_class)
    return num_correct.item()/torch.sum(mask).item() 

def validation(model, x, y, sequence_lengths, batch_size):
    session_size = x.shape[1]
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, sequence_lengths, None)
    output = output.view(batch_size, url_set_length, session_size) # according to pytorch CE api input format: (minibatch size, #classes, d)
    validation_accuracy = calc_accuracy(output, targets)
    del inputs 
    del targets
    del output
    return validation_accuracy

def train_minibatch(model, x, y, sequence_lengths, loss_function, optimizer, batch_size, url_set_length, device):
    session_size = x.shape[1]
    inputs = x.to(device)
    targets = y.to(device)
    output, _ = model(inputs, sequence_lengths, None)
    output = output.view(batch_size, url_set_length, session_size) # according to pytorch CE api input format: (minibatch size, #classes, d)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()

    train_accuracy = calc_accuracy(output, targets)
    del inputs 
    del targets
    del output
    return loss.item(), train_accuracy

def minibatch_create_train_and_validation(minibatch):
    train = minibatch[:-1]
    validation_set = [minibatch[-1]]
    train_inputs, train_labels, train_lengths = pad_minibatch(train)
    validation_inputs, validation_labels, validation_lengths = pad_minibatch(validation_set)
    return train_inputs, train_labels, train_lengths, validation_inputs, validation_labels, validation_lengths

def pad_minibatch(minibatch):
    sequences = [] 
    targets = []
    minibatch.sort(key=lambda x:len(x[0]), reverse=True)
    sequence_lengths = [len(s[0]) for s in minibatch]
    max_len = max(sequence_lengths)
    for seq, target in minibatch:
        pad = np.full((max_len - len(seq)),0)
        seq = np.concatenate((seq,pad)) 
        target = np.concatenate((target,pad)) 
        sequences.append(seq)
        targets.append(target)
    return torch.LongTensor(sequences), torch.LongTensor(targets), sequence_lengths
    

writer = SummaryWriter('logs') 
train_data, test_data, url_action_set_length, url_set_length = uap.create_dataset()

print('url action set length',url_action_set_length)
print('url set length',url_set_length)
print('training size',len(train_data))
print('test size',len(test_data))

n_iters = 1000
print_every = 5
test_every = 1
test_batch_size = 4
train_count = 0
train_loss_list = []
train_accuracies_list = []
validation_accuracies_list = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model(
    vocabulary_size=url_action_set_length,
    embedding_size=10,
    output_size=url_set_length, 
    device=device)

model.to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=0) # ignores zero-padding
optimizer = optim.Adam(model.parameters())

training_dataset = train_dataset.TrainDataset(train_data)
testing_dataset = test_dataset.TestDataset(test_data)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=5, collate_fn=minibatch_create_train_and_validation, shuffle=True, num_workers=4, drop_last=True)

start = time.time()
for i in range(1, n_iters + 1):
        
    for j, minibatch in enumerate(train_loader, 0):
        train_inputs, train_labels, train_lengths, validation_inputs, validation_labels, validation_lengths = minibatch
        
        # training
        training_loss, train_accuracy = train_minibatch(
            model=model, 
            x=train_inputs, 
            y=train_labels, 
            sequence_lengths=train_lengths,
            loss_function=loss_function, 
            optimizer=optimizer, 
            batch_size=train_inputs.shape[0], 
            url_set_length=url_set_length, 
            device=device)

        writer.add_scalar('Train/Loss', training_loss, train_count)
        writer.add_scalar('Train/Accuracy', train_accuracy, train_count)
        train_accuracies_list.append(train_accuracy)
        train_loss_list.append(training_loss)

        # validation 
        validation_accuracy = validation(
            model=model,
            x=validation_inputs,
            y=validation_labels,
            sequence_lengths=validation_lengths,
            batch_size=validation_inputs.shape[0]
        )

        writer.add_scalar('Validation/Accuracy', validation_accuracy, train_count)
        validation_accuracies_list.append(validation_accuracy)

        train_count += 1

    if i % print_every == 0:
        print('%s (%d %d%%)' % (timeSince(start), i, i / n_iters * 100))

# Test accuracy
testing_dataset = test_dataset.TestDataset(test_data)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=test_batch_size, collate_fn=pad_minibatch, shuffle=False, num_workers=4, drop_last=True)
test_accuracies_list = []
for j, minibatch in enumerate(test_loader, 0):
    x, y, sequence_lenghts = minibatch
    t_acc = test_accuracy(
        model=model, 
        x=x, 
        y=y, 
        sequence_lengths=sequence_lenghts,
        batch_size=test_batch_size, 
        url_set_length=url_set_length, 
        device=device)
    test_accuracies_list.append(t_acc)

print('test accuracy:',sum(test_accuracies_list)/len(test_accuracies_list))
plt.figure()
plt.plot(train_accuracies_list)
plt.savefig('img/train_acc.png')
plt.plot(train_loss_list)
plt.savefig('img/loss.png')
plt.plot(validation_accuracies_list)
plt.savefig('img/validation_acc.png')


  