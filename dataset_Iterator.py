import torch

class DataSetIterator(object):

    def __init__(self, batch_size, batchs, device):
        self.batch_size = batch_size
        self.batchs = batchs
        self.device = device
        self.batch_num = len(batchs) // batch_size
        self.residue = False
        if len(batchs) % batch_size != 0:
            self.residue = True
        self.index = 0
    
    def _to_tensor(self, batch):
        x = torch.LongTensor([_[0] for _ in batch]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch]).to(self.device)
        return x, y
    def __next__(self):
        if self.index == self.batch_num and self.residue:
            batch = self.batchs[self.index * batch_size:]
            self.index += 1
            batch = self._to_tensor(batch)
            return batch
        elif self.index >= self.batch_num:
            self.index = 0
            raise StopIteration
        else:
            batch = self.batchs[self.index * batch_size: (self.index + 1) * batch_size]
            self.index += 1
            batch = self._to_tensor(batch)
            return batch

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.batch_num + 1
        else:
            return self.batch_num

