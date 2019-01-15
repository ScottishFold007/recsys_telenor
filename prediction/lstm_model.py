import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, lstm_units, output_size, device):
        super(Model, self).__init__()
        self.hidden_dim = lstm_units
        self.device = device 
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, 
            embedding_dim=embedding_size, 
            padding_idx=0 # ignores zero-padding
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            batch_first=True # dim is: batch * sequence length * features
        )
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_size)
        self.softmax = nn.Softmax(dim=2)


    def init_hidden(self, minibatch_size):
        # the first is the hidden h
        # the second is the cell  c
        h = autograd.Variable(torch.zeros(2, minibatch_size, self.hidden_dim)).to(self.device)
        c = autograd.Variable(torch.zeros(2, minibatch_size, self.hidden_dim)).to(self.device)
        return (h,c)

    def forward(self, session, sequence_lengths, hidden):
        if hidden==None:
            hidden = self.init_hidden(session.shape[0])
        embedding = self.embedding(session)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, sequence_lengths, batch_first=True)

        lstm, hidden = self.lstm(packed, hidden) 
        
        pad_packed, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm, batch_first=True)
        
        linear = self.linear(pad_packed)

        output = self.softmax(linear)
        return output, hidden

