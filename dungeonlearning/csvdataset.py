import torch.utils.data as tdata
from torch import from_numpy
from numpy import genfromtxt, expand_dims

class CSVDataset(tdata.Dataset):
    def __init__(self, path, pattern, noFiles):
        super(CSVDataset, self).__init__()

        self.dungeonMaps = [];
        
        for fileNo in range(0, noFiles):
            filename = str.format("{}/{}{}.csv", path, pattern, fileNo);
            dungeonMap = genfromtxt(filename, delimiter=',')
            self.dungeonMaps.append(dungeonMap);

    def __len__(self):

        return len(self.dungeonMaps)

    def __getitem__(self, n):
        return from_numpy(expand_dims(self.dungeonMaps[n], axis=0)).float()

