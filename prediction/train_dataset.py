from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x,y = self.x_train[idx], self.y_train[idx]
        return x,y

