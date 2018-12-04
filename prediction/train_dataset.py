from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, train_data):
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        x,y = self.train_data[idx]
        return x,y

