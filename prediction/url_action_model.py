import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, output_size, minibatch_size, sequence_length, device):
        super(Model, self).__init__()
        #self.hidden_size = hidden_size
        self.hidden_dim = embedding_size
        self.minibatch_size = minibatch_size
        self.device = device 
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, num_layers=2)
        self.linear = nn.Linear(in_features=embedding_size, out_features=output_size)
        self.softmax = nn.Softmax(dim=0)
        self.hidden = self.init_hidden()  



    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        h = autograd.Variable(torch.zeros(2, self.minibatch_size, self.hidden_dim)).to(self.device)
        c = autograd.Variable(torch.zeros(2, self.minibatch_size, self.hidden_dim)).to(self.device)
        return (h,c)

    def forward(self, session, hidden):
        if hidden==None:
            hidden = self.hidden
        embedding = self.embedding(session)
        x = embedding.view(self.sequence_length, self.minibatch_size, -1) # according to pytorch api lstm input format: (seq len, minibatch size, features) 
        lstm, hidden = self.lstm(x,hidden) 
        linear = self.linear(lstm)
        output = self.softmax(linear.squeeze())
        return output, hidden

