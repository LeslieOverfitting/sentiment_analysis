import torch

class DataSet(object):
    def __init__(self, input_ids, input_masks, token_type_ids, labels, config):
        super().__init__()
        self.dataset = None
        self.build_input(input_ids, input_masks, token_type_ids, config)
        if labels is None:
            self.labels = torch.rand(self.dataset.size()[0])
        else:
            self.labels = torch.tensor(labels.astype(int) + 1, dtype=torch.long, device=config.device)
        
    def build_input(self, input_ids, input_masks, token_type_ids, config):
        self.dataset = torch.tensor(data=[input_ids, input_masks, token_type_ids], device=config.device).permute(1, 0, 2)