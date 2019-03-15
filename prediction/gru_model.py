import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, gru_layers, tied, dropout):
        super(Model, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers, dropout=dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, vocab_size)
        self.hidden_dim = hidden_dim

    def get_embedded(self, word_indexes):
        if self.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embedding(word_indexes)

    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        #out_packed_sequence, _ = self.gru(embedded_sents)
        out_packed_sequence, _ = self.lstm(embedded_sents)
        out = self.fc1(out_packed_sequence.data)

        return F.log_softmax(out, dim=1)