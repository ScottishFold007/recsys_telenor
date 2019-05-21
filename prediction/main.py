#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Trains a RNN model for next interaction prediction.
Combines a batched, non-padded variable length data input with torch.nn.Embedding.
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
import numpy as np
from model import Model

def load_dataset():
    #d = pickle.load( open( "./data/long_sessions_2.p", "rb" ) )
    #d = pickle.load( open( "./data/long.p", "rb" ) )
    #d = pickle.load( open( "./data/prepared_dataset_for_pretrained_emb.p", "rb" ) )
    d = pickle.load( open( "data/prepared_dataset.p", "rb" ) )
    return d['x_train'], d['vocab'] #, d['pre_trained_embeddings']


def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


def training_step(model, sents, loss_func, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = loss_func(out, y.data)
    return out, loss, y

def test_step(model, sents, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    return out, y

def calc_accuracy(output_distribution, targets):
    prediction = torch.argmax(output_distribution, dim=1)
    num_correct_prediction = (prediction == targets).sum()
    return num_correct_prediction.item()/targets.shape[0]

def topk_accuracy(output_distribution, targets, k):
    _, pred = torch.topk(input=output_distribution, k=k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.expand_as(pred))
    return correct.sum().item() / targets.shape[0]

def test_accuracy(test_data, model, args, device):
    model.eval()
    test_accuracies = []
    with torch.no_grad():
        for sents in batches(test_data, args.batch_size):
            out, y = test_step(model, sents, device)
            test_accuracies.append(calc_accuracy(out,y.data))
    return np.mean(test_accuracies)

def train_epoch(data, model, optimizer, loss_func, args, device):
    """ Trains a single epoch of the given model. """
    model.train()
    #log_timer = LogTimer(5)
    losses = []
    train_accuracies = []
    validation_accuracies = []
    train_perplexities = []
    precision = []
    for batch_ind, sents in enumerate(batches(data, args.batch_size)):
        taining_sents, validation_sents = split_minibatch(sents,args.validation_split_ratio)

        model.zero_grad()
        out, loss, y = training_step(model, taining_sents, loss_func, device)
        loss.backward()
        optimizer.step()

        # accuracy
        training_acc = calc_accuracy(out,y.data)
        p = topk_accuracy(out,y.data,k=4)
        validation_acc = validation(validation_sents, model, loss_func, device)

        # Calculate perplexity.
        prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
        perplexity = 2 ** prob.log2().neg().mean().item()
        logging.info("\tBatch %d, loss %.3f, perplexity %.2f training accuracy %.2f average validation accuracy %.2f",
                        batch_ind, loss.item(), perplexity, training_acc, validation_acc)
        losses.append(loss.item())
        train_accuracies.append(training_acc)
        validation_accuracies.append(validation_acc)
        train_perplexities.append(perplexity)
        precision.append(p)
    return losses, train_accuracies, validation_accuracies, train_perplexities, precision

def validation(validation_data, model, loss_func, device):
    model.eval()
    with torch.no_grad():
        out, y = test_step(model, validation_data, device)
        validation_accuracy = calc_accuracy(out,y.data)
        model.train()
        return validation_accuracy


def evaluate(evaluation_data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(evaluation_data, batch_size):
            out, y = test_step(model, sents, device)
            prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)

def split_minibatch(minibatch,split_ratio):
    train_size = int(split_ratio * len(minibatch))
    return minibatch[0:train_size], minibatch[train_size:-1]

def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--embedding-dim", type=int, default=20,
                      help="Word embedding dimensionality")
    argp.add_argument("--gru-hidden", type=int, default=100,
                      help="GRU gidden unit dimensionality")

    argp.add_argument("--gru-layers", type=int, default=1,
                      help="Number of GRU layers")
    argp.add_argument("--gru-dropout", type=float, default=0.0,
                      help="The amount of dropout in GRU layers")
    argp.add_argument("--validation-split-ratio", type=float, default=0.8,
                      help="percentage of minibatch used for training")

    argp.add_argument("--epochs", type=int, default=20)
    argp.add_argument("--batch-size", type=int, default=200)
    argp.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate")

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    #device = torch.device("cpu")

    t_size = int(args.validation_split_ratio * args.batch_size)
    v_size = args.batch_size - t_size


    train_data, vocab = load_dataset() 
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    
    model = Model(214, args.embedding_dim, args.gru_hidden, args.gru_layers, args.gru_dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    loss_per_batch = []
    training_accuracy_per_batch = []
    validation_accuracy_per_batch = []
    training_perplexities_per_batch = []
    training_precission_per_batch = []

    
    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)
        l, train_accuracy, validation_accuracy, p, prec = train_epoch(train_data, model, optimizer, loss_func, args, device)
        loss_per_batch.extend(l)
        training_accuracy_per_batch.extend(train_accuracy)
        validation_accuracy_per_batch.extend(validation_accuracy)
        training_perplexities_per_batch.extend(p)
        training_precission_per_batch.extend(prec)

    torch.save(model.state_dict(), './state_dict.pth')

    '''
    params = list(model.parameters())
    with open('embeddings.p', 'wb') as f:
        pickle.dump((params[0].data, vocab), f)
    
    del model
    '''

    # ploting
    plt.figure()
    plt.plot(loss_per_batch)
    plt.savefig('img/loss_per_batch.pdf')
    plt.figure()
    plt.plot(training_perplexities_per_batch)
    plt.savefig('img/training_perplexities_per_batch.pdf')
    plt.figure()
    plt.plot(training_accuracy_per_batch)
    plt.savefig('img/training_accuracy_per_batch.pdf')
    plt.figure()
    plt.plot(validation_accuracy_per_batch)
    plt.savefig('img/validation_accuracy_per_batch.pdf')
    plt.figure()
    plt.plot(training_precission_per_batch)
    plt.savefig('img/training_precission_per_batch.pdf')

    # save results 
    log_dict = {}
    log_dict['loss_per_batch'] = loss_per_batch
    with open('logs/action.p', 'wb') as f:
        pickle.dump(log_dict, f)
    
    


if __name__ == '__main__':
    main()
