import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, url_set_length, action_set_length):
        super(Model, self).__init__()
        #self.hidden_size = hidden_size

        self.embedding_1 = nn.Embedding(num_embeddings=100, embedding_dim=action_set_length)
        self.embedding_2 = nn.Embedding(num_embeddings=100, embedding_dim=action_set_length)
        self.lstm = nn.LSTM(input_size=action_set_length, hidden_size=action_set_length, num_layers=2, dropout=0.05)
        self.linear = nn.Linear(in_features=5320, out_features=28)
        self.softmax = nn.Softmax()

    def forward(self, x1, x2):
        embedding_1 = self.embedding_1(x1)
        embedding_2 = self.embedding_2(x2)
        concat = torch.cat((embedding_1.data, embedding_2.data))
        concat = concat.unsqueeze(1)
        lstm, hidden = self.lstm(concat)
        flat = lstm.squeeze(0)
        flat2 = flat.view(flat.numel())
        #print(lstm.shape)
        linear = self.linear(flat2)
        print(linear)
        output = self.softmax(linear.unsqueeze(1))
        print(output.shape)
        print(output)
        return output