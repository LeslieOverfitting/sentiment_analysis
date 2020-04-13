import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch


class ErnieModel_Base(nn.Module):
    def __init__(self, bert_model_path, config, seq_len):
        super(ErnieModel_Base, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_model_path, config=config)
        self.linear_1 = nn.Linear(in_features=seq_len, out_features=1, bias=True)
        self.linear_2 = nn.Linear(in_features=config.hidden_size, out_features=3, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.linear_1.apply(self.init_network)
        self.linear_2.apply(self.init_network)
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False
        

    def forward(self, inputs):
        sequence_output, pooler_output, hidden_states  = self.bert_model(input_ids=inputs[:, 0], attention_mask=inputs[:, 1], token_type_ids=inputs[:, 2])
        outputs = self.linear_1(sequence_output.permute(0, 2, 1)).squeeze(dim=2)
        # (batch_size, sequence_length, hidden_size)-> (batch_size, hidden_size)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear_2(outputs)  # (batch_size, hidden_size)->(batch_size,3)
        outputs = F.softmax(outputs, dim=1)
        return outputs

    def init_network(self):
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.xavier_normal_(self.linear_2.weight)

    def init_network(self, module):
        if isinstance(module, nn.Linear):
          print(module.__class__.__name__)
          nn.init.xavier_uniform_(module.weight.data)
          nn.init.constant_(module.bias.data, 0.0)