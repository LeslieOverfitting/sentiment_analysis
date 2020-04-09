import torch

class DataSet(object):
    def __init__(self, input_ids, input_masks, token_type_ids, labels, device):
        super().__init__()
        self.dataset = None
        self.build_input(input_ids, input_masks, token_type_ids, device)
        if labels is None:
            self.labels = torch.rand(self.dataset.size()[0], device=device)
        else:
            self.labels = torch.tensor(labels.astype(int) + 1, dtype=torch.long, device=device)
        
    def build_input(self, input_ids, input_masks, token_type_ids, device):
        self.dataset = torch.tensor(data=[input_ids, input_masks, token_type_ids], device=device).permute(1, 0, 2)