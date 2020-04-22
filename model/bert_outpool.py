import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch


class Bert_outpool(nn.Module):
    def __init__(self, bert_model_path, config, seq_len):
        super(Bert_outpool, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_model_path, config=config)
        self._predict_fc = nn.Sequential(
                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True),
                          nn.Tanh(),
                          nn.Dropout(p=0.2),
                          nn.Linear(in_features=config.hidden_size, out_features=3, bias=True)
                        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(6)])
        self._predict_fc.apply(self.init_model_weights)

    def forward(self, inputs):
        sequence_output, pooler_output, hidden_states  = self.bert_model(input_ids=inputs[:, 0], attention_mask=inputs[:, 1], token_type_ids=inputs[:, 2])
        # (batch_size, sequence_length, hidden_size)-> (batch_size, hidden_size)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(pooler_output)
                out = out.view(out.size(0), -1)
                out = self._predict_fc(out)
            else:
                temp_out = dropout(pooler_output)
                temp_out = temp_out.view(out.size(0), -1)
                out = out + self._predict_fc(temp_out)
        return out / len(self.dropouts)

    def init_model_weights(self, module):
        """
        Initialise the weights of the inferSent model.
        """
        if isinstance(module, nn.Linear):
            print(module.__class__.__name__)
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)