from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self, test_data):
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        x,y = self.test_data[idx]
        return x,y

