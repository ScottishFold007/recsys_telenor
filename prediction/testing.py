import pickle
import torch
import torch.nn as nn
import random
import numpy as np
from model import Model

def load_dataset():
    #d = pickle.load( open( "./data/short_sessions.p", "rb" ) )
    d = pickle.load( open( "./data/prepared_dataset.p", "rb" ) )
    #d = pickle.load( open( "./data/prepared_dataset_for_pretrained_emb.p", "rb" ) )
    return d['x_test'], d['vocab'] #, d['pre_trained_embeddings']

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
    print(len(x_test))
    model = Model(vocab_size=214, embedding_dim=20, hidden_dim=100, gru_layers=1, dropout=0.0)
    batch_size = 200
    model.load_state_dict(torch.load('./state_dict.pth'))
    model.eval()
    model.to(device)
    test_accuracies = []
    top2_accuracies = []
    top3_accuracies = []
    with torch.no_grad():
        for sents in batches(x_test, batch_size):
            out, y = step(model, sents, device)
            test_accuracies.append(calc_accuracy(out,y.data))
            top2_accuracies.append(topk_accuracy(out,y.data, 2))
            top3_accuracies.append(topk_accuracy(out,y.data, 3))
    print('test accuracy', np.mean(test_accuracies))
    print('test top 2 accuracy', np.mean(top2_accuracies))
    print('test top 3 accuracy', np.mean(top3_accuracies))
test_accuracy()