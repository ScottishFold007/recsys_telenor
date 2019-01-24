from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x,y = self.x_test[idx], self.y_test[idx]
        return x,y

