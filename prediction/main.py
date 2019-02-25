#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Trains and evaluates an RNN language model written using
PyTorch v0.4. Illustrates how to combine a batched, non-padded
variable length data input with torch.nn.Embedding and how to
use tied input and output word embeddings.
"""

from argparse import ArgumentParser
import logging
import math
import random
import sys
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import url_action_preparation as uap 
import train_dataset 
import test_dataset
from gru_model import Model


def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


def step(model, sents, loss_func, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    #F.nll_loss(out, y.data)
    loss = loss_func(out, y.data)
    return out, loss, y

def calc_accuracy(output_distribution, targets):
    prediction = torch.argmax(output_distribution, dim=1)
    mask = targets != 0
    correct_with_padding = prediction == targets
    correct_class = mask * correct_with_padding
    num_correct = torch.sum(correct_class)
    return num_correct.item()/torch.sum(mask).item() 


def train_epoch(data, model, optimizer, loss_func, args, device):
    """ Trains a single epoch of the given model. """
    model.train()
    #log_timer = LogTimer(5)
    losses = []
    train_accuracies = []
    for batch_ind, sents in enumerate(batches(data, args.batch_size)):
        taining_sents, validation_sents = split_minibatch(sents)

        validation_acc = validation(validation_sents, model, args.batch_size, loss_func, device)

        model.zero_grad()
        out, loss, y = step(model, taining_sents, loss_func, device)
        loss.backward()
        optimizer.step()

        # accuracy
        acc = calc_accuracy(out,y.data)

        # Calculate perplexity.
        prob = out.exp()[
            torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
        perplexity = 2 ** prob.log2().neg().mean().item()
        logging.info("\tBatch %d, loss %.3f, perplexity %.2f training accuracy %.2f average validation accuracy %.2f",
                        batch_ind, loss.item(), perplexity, acc, validation_acc)
        losses.append(loss.item())
        train_accuracies.append(acc)
    return losses, train_accuracies

def validation(data, model, batch_size, loss_func, device):
    model.eval()
    with torch.no_grad():
        acc_sum = 0.0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, loss_func, device)
            acc_sum += calc_accuracy(out,y.data)
        model.train()
        return acc_sum/len(data)

def evaluate(data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, device)
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)

def split_minibatch(minibatch):
    train_size = int(0.8 * len(minibatch))
    return minibatch[0:train_size], minibatch[train_size:-1]

def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--embedding-dim", type=int, default=512,
                      help="Word embedding dimensionality")
    argp.add_argument("--untied", action="store_true",
                      help="Use untied input/output embedding weights")
    argp.add_argument("--gru-hidden", type=int, default=512,
                      help="GRU gidden unit dimensionality")
    argp.add_argument("--gru-layers", type=int, default=1,
                      help="Number of GRU layers")
    argp.add_argument("--gru-dropout", type=float, default=0.0,
                      help="The amount of dropout in GRU layers")

    argp.add_argument("--epochs", type=int, default=20)
    argp.add_argument("--batch-size", type=int, default=128)
    argp.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate")

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    #device = torch.device("cpu")


    x_train, y_train, x_test, y_test, input_vocabulary_size, target_vocabulary_size, vocab = uap.create_dataset_action()
    train_data = x_train#[(x,y) for x,y in zip(x_train,y_train)]
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    '''
    vocab = Vocab()
    
    # Load data now to know the whole vocabulary when training model.
    train_data = data_loader.load(data_loader.path("train"), vocab)
    valid_data = data_loader.load(data_loader.path("valid"), vocab)
    test_data = data_loader.load(data_loader.path("test"), vocab)
    '''
    model = Model(len(vocab), args.embedding_dim,
                  args.gru_hidden, args.gru_layers,
                  not args.untied, args.gru_dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)

    loss_per_batch = []
    loss_per_epoch = []


    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)
        l, a = train_epoch(train_data, model, optimizer, loss_func, args, device)
        loss_per_batch.extend(l)
        loss_per_epoch.append(sum(l)/len(l))
    
    # ploting
    plt.figure()
    plt.plot(loss_per_batch)
    plt.savefig('img/loss_per_batch.pdf')
    plt.figure()
    plt.plot(loss_per_epoch)
    plt.savefig('img/loss_per_epoch.pdf')

    # save results 
    log_dict = {}
    log_dict['loss_per_batch'] = loss_per_batch
    with open('logs/action.p', 'wb') as f:
        pickle.dump(log_dict, f)


if __name__ == '__main__':
    main()
