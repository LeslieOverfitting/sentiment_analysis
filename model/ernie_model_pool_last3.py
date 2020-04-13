import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch


class Ernie_PoolLast3_Model(nn.Module):
    def __init__(self, bert_model_path, config, seq_len):
        super(Ernie_PoolLast3_Model, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_model_path, config=config)
        self._predict_fc = nn.Sequential(nn.Dropout(p=0.2),
                                         nn.Linear(in_features=config.hidden_size * 4, out_features=config.hidden_size, bias=True),
                                         nn.Tanh(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(in_features=config.hidden_size, out_features=3, bias=True)
                                        )
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False
        self._predict_fc.apply(self.init_model_weights)

    def forward(self, inputs):
        sequence_output, pooler_output, hidden_states  = self.bert_model(input_ids=inputs[:, 0], attention_mask=inputs[:, 1], token_type_ids=inputs[:, 2])
        # (batch_size, sequence_length, hidden_size)-> (batch_size, hidden_size)
        inputs = torch.cat([pooler_output, hidden_states[-1][:,0], hidden_states[-2][:,0], hidden_states[-2][:,0]], dim = 1)
        predicts = self._predict_fc(inputs) 
        return predicts

    def init_model_weights(self, module):
        """
        Initialise the weights of the inferSent model.
        """
        if isinstance(module, nn.Linear):
            print(module.__class__.__name__)
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)