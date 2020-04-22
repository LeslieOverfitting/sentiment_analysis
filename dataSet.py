import torch
from torch.utils.data import Dataset

class BertDataSet(Dataset):
    def __init__(self, input_ids, input_masks, token_type_ids, labels=None, device='cpu'):
        super().__init__()
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.token_type_ids = token_type_ids
        self.labels = labels
   
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.input_ids[idx], self.input_masks[idx], self.token_type_ids[idx]
        return self.input_ids[idx], self.input_masks[idx], self.token_type_ids[idx], self.labels[idx]