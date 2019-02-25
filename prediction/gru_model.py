import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers, tied, dropout):
        super(Model, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

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
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, _ = self.gru(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)