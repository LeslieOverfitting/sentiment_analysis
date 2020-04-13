import torch
import torch.nn as nn
import torch.nn.functional as F

class STModel(nn.Module):
    
    def __init__(self, emb_dim, vocab_size, classes_num, hidden_num, layer_num, device, embeddings = None, padding_idx = 0, dropout=0.5):
        super(STModel, self).__init__()
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device
        self.bidirectiona = True
        self.encoder = nn.Embedding(vocab_size, emb_dim, padding_idx= padding_idx)
        if embeddings is not None:
            self.encoder.weight.data.copy_(embeddings)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_num, layer_num, dropout= dropout,
                            bidirectional= self.bidirectiona, batch_first = True)
        
        self.attention_query_layer = nn.Sequential(
            nn.Linear(self.hidden_num, self.hidden_num),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(hidden_num, classes_num)
        self.init_weights()
        self.sigm = nn.Sigmoid()
        
        
    def forward(self, x, hidden, print_flag = False):
        batch_size = x.size(0)
        if(print_flag):
          print(batch_size)
        emb = self.drop(self.encoder(x))
        if(print_flag):
          print('emb', emb.size())
        if(print_flag):
          print('hidden', hidden)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(emb, hidden)
        if(print_flag):
          print('lstm_out', lstm_out.size())
        lstm_out = self.drop(lstm_out) 
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        fc_input = self.attention_net(lstm_out, final_hidden_state)
        if(print_flag):
          print('fc_input', fc_input.size())
        fc_out = self.fc(fc_input)
        return fc_out
    
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        for layer in self.attention_query_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
    
    def attention_net(self, lstm_out, lstm_hidden):
        """
            lstm_out: [batch, seq_len , (hidden_num * direction)]
            lstm_hidden: [batch , (direction * layer_num) , hidden_num]
            return [batch, hidden_num]
            https://blog.csdn.net/dendi_hust/article/details/94435919?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
        """
        # generate query by lstm_hidden
        lstm_hidden = torch.sum(lstm_hidden, dim= 1) # [batch , hidden_num]
        lstm_hidden = lstm_hidden.unsqueeze(1) # [batch ,1 ,hidden_num]
        atten_query = self.attention_query_layer(lstm_hidden) # [batch ,1 ,hidden_num]

        lstm_out = torch.chunk(lstm_out, self.layer_num, -1) # 默认先是两层
        lstm_value = lstm_out[0] + lstm_out[1] # [batch, seq_len , hidden_num]
        lstm_value_tanh = m = nn.Tanh()(lstm_value) # [batch, seq_len , hidden_num]
        #print("lstm_value_tanh" ,lstm_value_tanh.size())
        #print("atten_query" ,atten_query.size())
        # calculate weight
        atten_weight = torch.bmm(atten_query, lstm_value_tanh.transpose(1,2)) # [batch, 1 , seq_len]
        softmax_w = F.softmax(atten_weight, dim=-1)# [batch, 1 , seq_len]
        atten_value = torch.bmm(softmax_w, lstm_value_tanh) # [batch, 1 , hidden_num]
        result = atten_value.squeeze(1)
        return result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        num = 1
        if self.bidirectiona:
          num = 2
        if(self.device):
            return (weight.new_zeros(self.layer_num * num, batch_size, self.hidden_num).cuda(),
                    weight.new_zeros(self.layer_num * num, batch_size, self.hidden_num).cuda())
        else:
            return (weight.new_zeros(self.layer_num, batch_size, self.hidden_num),
                    weight.new_zeros(self.layer_num, batch_size, self.hidden_num))