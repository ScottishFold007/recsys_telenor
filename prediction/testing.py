import pickle
import torch
import torch.nn as nn
import random
import numpy as np
from gru_model import Model

def load_dataset():
    d = pickle.load( open( "./prepared_dataset.p", "rb" ) )
    return d['x_test'], d['vocab']

def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]

def step(model, sents, device):
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
    num_correct_prediction = (prediction == targets).float().sum()
    return num_correct_prediction.item()/targets.shape[0]

def topk_accuracy(output_distribution, targets, k):
    _, pred = torch.topk(input=output_distribution, k=k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.expand_as(pred))
    return correct.sum().item() / targets.shape[0]

def test_accuracy():
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    x_test, vocab = load_dataset()
    model = Model(len(vocab), 100, 100, 1, not '--untied', 0.0)
    model.load_state_dict(torch.load('./state_dict.pth'))
    model.eval()
    model.to(device)
    test_accuracies = []
    topk_accuracies = []
    k = 2
    with torch.no_grad():
        for sents in batches(x_test, 100):
            out, y = step(model, sents, device)
            test_accuracies.append(calc_accuracy(out,y.data))
            topk_accuracies.append(topk_accuracy(out,y.data, k))
    print('test accuracy', np.mean(test_accuracies))
    print('test top K accuracy', np.mean(topk_accuracies))
test_accuracy()