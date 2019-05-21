import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """ A RNN model for next interaction prediction. """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, gru_layers, dropout):
        super(Model, self).__init__()
        #self.embedding = self.embedding, embedding_dim = self.create_emb_layer(pre_trained_embeddings, True) 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.recurrent_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers, dropout=dropout, bidirectional=True)
        self.fully_connected = nn.Linear(hidden_dim*2, vocab_size)
    
    def create_emb_layer(self, pre_trained_embeddings, non_trainable=False):
        num_embeddings, embedding_dim = pre_trained_embeddings.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': torch.from_numpy(pre_trained_embeddings)})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, embedding_dim


    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(self.embedding(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, hidden = self.recurrent_layer(embedded_sents)
        out = self.fully_connected(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)