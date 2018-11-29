import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):
    def __init__(self, url_action_set_length, url_set_length):
        super(Model, self).__init__()
        #self.hidden_size = hidden_size
        self.hidden_dim = url_action_set_length
        self.embedding = nn.Embedding(num_embeddings=url_action_set_length, embedding_dim=url_action_set_length)
        self.lstm = nn.LSTM(input_size=url_action_set_length, hidden_size=url_action_set_length, num_layers=2)
        self.linear = nn.Linear(in_features=url_action_set_length, out_features=url_set_length)
        self.softmax = nn.Softmax(dim=0)
        self.hidden = self.init_hidden()  



    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)), autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))

    def forward(self, url_action, hidden):
        if hidden==None:
            hidden = self.hidden
        embedding = self.embedding(url_action)
        lstm, hidden = self.lstm(embedding.unsqueeze(0).unsqueeze(0),hidden) 
        linear = self.linear(lstm)
        output = self.softmax(linear.squeeze())
        return output.unsqueeze(0), hidden