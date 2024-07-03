from typing import Optional, Tuple

import torch
from tensorflow import Tensor
from torch import nn


# Embedded [batch_size, seq_len, input_dimension_reflected_to]



class LSTMMultiheadAttention(nn.Module):
    def __init__(self,
                 batch_size: int,
                 feature_size: int,
                 num_heads: int,
                 number_of_layers: int,
                 lstm_hidden_dim: int,
                 dropout: float,
                 bias: bool,
                 dtype: torch.dtype,
                 device: torch.device, ):
        """
        # @Input: [seq_len, batch_size, embed_dim] \n
        # @Output: [seq_len, batch_size, embed_dim]



        :param batch_size:
        :param feature_size:
        :param num_heads:
        :param lstm_hidden_dim:
        :param dropout:
        :param bias:
        :param dtype:
        :param device:
        """
        """

                                        ↓
        +----------------------------------------------------------------+
        |                      Parallel LSTM Layers                      |
        +----------------------------------------------------------------+
        |         Q          |          K          |          V          |
        +----------------------------------------------------------------+
                                        ↓
        +----------------------------------------------------------------+
        |                      Multi-head Attention                      |
        +----------------------------------------------------------------+
        |          Q          |          K         |          V          |
        +----------------------------------------------------------------+
        |  +-----------------------------------------------------------+ |
        |  |              Original Linear Transformation               | |
        |  +-----------------------------------------------------------+ |
        +----------------------------------------------------------------+
        |         WQ         |         WK         |          WV          |
        +----------------------------------------------------------------+
        |  +----------------------------------------------------------+  |
        |  |                   Attention Calculation                  |  |
        |  +----------------------------------------------------------+  |
        +----------------------------------------------------------------+
        |   Attention(Q, K, V) = softmax((WQ * WK^T) / sqrt(d_k)) * WV   |
        +----------------------------------------------------------------+
        |  +----------------------------------------- -----------------+ |
        |  |                           Output                          | |
        |  +-----------------------------------------------------------+ |
        +----------------------------------------------------------------+
                                        ↓
        """
        super(LSTMMultiheadAttention, self).__init__()
        # @Input: [seq_len, batch_size, embed_dim]
        # 使用LSTM层替代原有的线性层
        '''
        input(seq_len,batch,input_size)
        h0(num_layers*num_directions,batch,hidden_size)
        c0(num_layers*num_directions,batch,hidden_size)
        output(seq_len,batch,num_direction*hidden_size)
        '''

        # ================================== Parameter Backup ========================================
        self.data_type: torch.dtype = dtype
        self.device: torch.device = device

        self.batch_size: int = batch_size
        self.feature_size: int = feature_size
        self.num_heads: int = num_heads
        self.number_of_layers: int = number_of_layers
        self.lstm_hidden_dim: int = lstm_hidden_dim
        self.dropout: float = dropout
        self.bias: bool = bias


        self.number_of_layers: int = number_of_layers

        self.lstm_query = nn.LSTM(num_layers=number_of_layers,
                                  input_size=feature_size,
                                  hidden_size=lstm_hidden_dim,
                                  device=self.device)
        self.lstm_key = nn.LSTM(num_layers=number_of_layers,
                                input_size=feature_size,
                                hidden_size=lstm_hidden_dim,
                                device=self.device)
        self.lstm_value = nn.LSTM(num_layers=number_of_layers,
                                  input_size=feature_size,
                                  hidden_size=lstm_hidden_dim,
                                  device=self.device)

        # [num_layers*num_directions, batch, hidden_size]
        self.lstm_query_cell_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=device)
        self.lstm_key_cell_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                             dtype=self.data_type).to(device=device)
        self.lstm_value_cell_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=device)

        # [num_layers*num_directions, batch, hidden_size]
        self.lstm_query_hidden_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                                 dtype=self.data_type).to(device=device)
        self.lstm_key_hidden_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=device)
        self.lstm_value_hidden_state: torch.Tensor = torch.randn(number_of_layers, batch_size, lstm_hidden_dim,
                                                                 dtype=self.data_type).to(device=device)

        '''
        Q: seq_len1, batch_size, num_directions
        K: seq_len2, batch_size, num_directions
        V: seq_len2, batch_size, num_directions
        output: seq_len, batch_size, num_directions
        attention batch_size, seq_len1, seq_len2
        '''
        self.multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
            device=device
        )

    def reinit_hidden_cell_state(self):
        # [num_layers*num_directions, batch, hidden_size]
        self.lstm_query_cell_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                               self.batch_size,
                                                               self.lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=self.device)
        self.lstm_key_cell_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                             self.batch_size,
                                                             self.lstm_hidden_dim,
                                                             dtype=self.data_type).to(device=self.device)
        self.lstm_value_cell_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                               self.batch_size,
                                                               self.lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=self.device)

        # [num_layers*num_directions, batch, hidden_size]
        self.lstm_query_hidden_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                                 self.batch_size,
                                                                 self.lstm_hidden_dim,
                                                                 dtype=self.data_type).to(device=self.device)
        self.lstm_key_hidden_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                               self.batch_size,
                                                               self.lstm_hidden_dim,
                                                               dtype=self.data_type).to(device=self.device)
        self.lstm_value_hidden_state: torch.Tensor = torch.randn(self.number_of_layers,
                                                                 self.batch_size,
                                                                 self.lstm_hidden_dim,
                                                                 dtype=self.data_type).to(device=self.device)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # 使用LSTM层计算Q, K, V

        # 解耦
        self.lstm_query_hidden_state = self.lstm_query_hidden_state.detach()
        self.lstm_query_cell_state = self.lstm_query_cell_state.detach()
        self.lstm_key_hidden_state = self.lstm_key_hidden_state.detach()
        self.lstm_key_cell_state = self.lstm_key_cell_state.detach()
        self.lstm_value_hidden_state = self.lstm_value_hidden_state.detach()
        self.lstm_value_cell_state = self.lstm_value_cell_state.detach()

        # [seq_len, batch, hidden_size]
        query, (self.lstm_query_hidden_state, self.lstm_query_cell_state) \
            = self.lstm_query(query,
                              (self.lstm_query_hidden_state,
                               self.lstm_query_cell_state))

        key, (self.lstm_key_hidden_state, self.lstm_key_cell_state) \
            = self.lstm_key(key,
                            (self.lstm_key_hidden_state,
                             self.lstm_key_cell_state))

        value, (self.lstm_value_hidden_state, self.lstm_value_cell_state) \
            = self.lstm_value(value,
                              (self.lstm_value_hidden_state,
                               self.lstm_value_cell_state))

        # Multihead Attention
        # [seq_len, batch, embed_dim]
        # => [seq_len, batch, embed_dim]
        output, attention = self.multihead_attention(query, key, value)

        # @output: [seq_len, batch, embed_dim]
        return output, attention
