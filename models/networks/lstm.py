import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential()
            self.bidirectional = bidirectional
            if bidirectional:
                self.dense_layer.add_module('linear', nn.Linear(2 * self.hidden_size, self.hidden_size))
            else:
                self.dense_layer.add_module('linear', nn.Linear(self.hidden_size, self.hidden_size))
            self.dense_layer.add_module('activate', nn.Tanh())
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文: Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        # sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        
        # return sentence_vector
        '''edit here (zelin)'''
        attended_r_out = r_out * atten_weight  # 保持 [batch_size, seq_len, hidden_size]
        return attended_r_out  # No sum over time dimension

    def embd_maxpool(self, r_out, h_n):
        
        """保留了时间维度，使用 torch.max 时增加 keepdim=True，并扩展结果"""
        pooled_out, _ = torch.max(r_out, dim=1, keepdim=True)  # Keeps time dim
        return pooled_out.expand_as(r_out)  # Duplicate across time dimension

    def embd_last(self, r_out, h_n):
        
        return r_out  # Returns [batch_size, seq_len, hidden_size]

    def embd_dense(self, r_out, h_n):
        '''
        每个时间步应用 dense_layer，然后将其还原为原始的三维格式 [batch_size, seq_len, hidden_size]。
        '''
        seq_len = r_out.size(1)
        r_out = r_out.reshape(-1, r_out.size(2))  # Flatten to [batch_size * seq_len, hidden_size]
        dense_out = self.dense_layer(r_out)
        return dense_out.reshape(-1, seq_len, self.hidden_size)  # Reshape back to [batch_size, seq_len, hidden_size]

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)
        return embd