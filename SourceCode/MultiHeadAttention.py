import profile
import typing
from typing import Any, Optional, Tuple
import warnings

import numpy
import torch
from tensorflow import Tensor
from torch import nn

import Toolbox
import gc

import memory_profiler

# Embedded [batch_size, seq_len, input_dimension_reflected_to]


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 batch_size: int,
                 head_number: int,
                 sequence_length: int,
                 input_dimension_reflected_to: int | float,
                 device: torch.device,
                 dtype: torch.dtype,
                 dimension_when_query_halfed: int | float = 16,
                 dimension_when_calc_key_halfed: int | float = 16,
                 dimension_when_calc_value_halfed: int | float = 16,
                 whether_RNN: str = "LSTM",
                 if_index: int | Any = 0,
                 mask_bool: bool = True,
                 bidirectional: bool = True
                 ) -> None:
        """
        dimension_when_calc_value_halfed = dimension_when_calc_key_halfed
        :param batch_size:
        :param head_number:
        :param sequence_length:
        :param input_dimension_reflected_to:
        :param device:
        :param dimension_when_query_halfed:
        :param dimension_when_calc_key_halfed:
        :param dimension_when_calc_value_halfed:
        :param dimension_reflected_to_out:
        :param whether_RNN:
        :param if_index:
        """
        super(MultiHeadAttention, self).__init__()

        self.dropout_rate: float = 0.1

        if type(if_index) is int:
            print("Entering MultiHeadAttention Init, ", if_index)
        else:
            pass

        if type(dimension_when_query_halfed) is int or str(dimension_when_query_halfed)[-2:] == ".5":
            print("Entering MultiHeadAttention Init, ", dimension_when_query_halfed)
        else:
            raise ValueError("dimension_when_query_halfed HAS to be Int or .5 Float")

        if type(dimension_when_calc_key_halfed) is int or str(dimension_when_calc_key_halfed)[-2:] == ".5":
            print("Entering MultiHeadAttention Init, ", dimension_when_calc_key_halfed)
        else:
            raise ValueError("dimension_when_calc_key_halfed HAS to be Int or .5 Float")

        if type(dimension_when_calc_value_halfed) is int or str(dimension_when_calc_value_halfed)[-2:] == ".5":
            print("Entering MultiHeadAttention Init, ", dimension_when_calc_value_halfed)
        else:
            raise ValueError("dimension_when_calc_value_halfed HAS to be Int or .5 Float")

        self.bidirectional: bool = bidirectional
        self.bidirectional_int: int = 2 if bidirectional else 1
        # 备份参数
        self.batch_size: int = batch_size
        self.head_number: int = head_number
        self.sequence_length: int = sequence_length
        self.input_dimension_reflected_to: int = input_dimension_reflected_to
        self.dimension_when_query_halfed: int = dimension_when_query_halfed
        self.dimension_when_calc_key_halfed: int = dimension_when_calc_key_halfed

        self.whether_RNN: str = whether_RNN

        self.dimension_when_calc_value_half: int = dimension_when_calc_value_halfed

        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        self.mask_bool: bool = mask_bool

        # 输入的Embedding矩阵 [batch_size, seq_len, input_dimension_reflected_to]
        self.total_layers_RNN: int = 8

        print("Finish MultiHeadAttention parameter backup")

        # ===================== Initialize Query, Key, Value Matrix =====================
        # --------------------- Query ---------------------
        if self.whether_RNN == "RNN":
            # 随机初始化一个W_{Q}矩阵(以RNN代替)
            # 目的是让网络具有记忆性，因为我过去经验（Attention）和现在的经验是有关联的
            # [input_dimension_reflected_to, dimension_when_query_halfed]
            # 创建批量处理的输入张量，形状为 (batch_size, sequence_length, input_size)
            # 创建批量处理的初始隐藏状态，形状为 (num_layers, batch_size, hidden_size)
            # 输出 (batch_size, sequence_length, hidden_size)
            # input_size = input_dimension_reflected_to/
            # hidden_size = dimension_when_query_halfed
            self.form_query_layer: nn.RNN = nn.RNN(input_size=input_dimension_reflected_to,
                                                   hidden_size=dimension_when_query_halfed * head_number,
                                                   num_layers=self.total_layers_RNN,
                                                   bidirectional=bidirectional,
                                                   batch_first=True,
                                                   bias=False,
                                                   dtype=dtype)

        elif self.whether_RNN == "LSTM":
            # 输入 (batch_size, seq_length, input_size)
            # 输出 (batch_size, seq_length, hidden_size)
            self.form_query_layer: nn.LSTM = nn.LSTM(
                input_size=input_dimension_reflected_to,
                hidden_size=dimension_when_query_halfed * head_number,
                num_layers=self.total_layers_RNN,
                bidirectional=bidirectional,
                batch_first=True,
                bias=False,
                dropout=self.dropout_rate,
                dtype=dtype
            )

        else:
            self.form_query_layer: nn.Linear = nn.Linear(input_dimension_reflected_to,
                                                         dimension_when_query_halfed * self.bidirectional_int * head_number,
                                                         bias=False,
                                                         dtype=dtype)

        print("Form Query Layer Finish")

        # $$$$$$$$$$$$$$$$$$$$$ For RNN Hidden State $$$$$$$$$$$$$$$$$$$$$
        if self.whether_RNN == "RNN":
            # RNN 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections,
            # batch_size, dimension_when_query_halfed],不受batch_first影响
            self.hidden_inout_form_query_layer: torch.Tensor = torch.randn(
                self.total_layers_RNN * self.bidirectional_int,
                batch_size,
                dimension_when_query_halfed * head_number)
        elif self.whether_RNN == "LSTM":
            # LSTM 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections,
            # batch_size, dimension_when_query_halfed],不受batch_first影响
            self.hidden_inout_form_query_layer: torch.Tensor = torch.randn(
                self.total_layers_RNN * self.bidirectional_int,
                batch_size,
                dimension_when_query_halfed * head_number)
            self.hidden_LSTM_cell_state: torch.Tensor = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                                                                    batch_size,
                                                                    dimension_when_query_halfed * head_number)
        else:
            pass

        print("----Finish Hidden Query Layer Tensor----")

        # --------------------- Key ---------------------
        if self.whether_RNN == "RNN":
            # 随机初始化一个W_{K}矩阵(以RNN代替)
            # 目的是让网络具有记忆性，因为我过去经验（Attention）和现在的经验是有关联的
            self.form_key_layer: nn.RNN = nn.RNN(input_size=input_dimension_reflected_to,
                                                 hidden_size=dimension_when_calc_key_halfed * head_number,
                                                 num_layers=self.total_layers_RNN,
                                                 bidirectional=bidirectional,
                                                 batch_first=True,
                                                 bias=False,
                                                 dtype=dtype)
        elif self.whether_RNN == "LSTM":
            # LSTM 需要一个hidden I/O hidden总是 [total_layers_RNN * NumberOfDirections, batch_size,
            # dimension_when_calc_key_halfed],不受batch_first影响
            self.form_key_layer: nn.LSTM = nn.LSTM(
                input_size=input_dimension_reflected_to,
                hidden_size=dimension_when_calc_key_halfed * head_number,
                num_layers=self.total_layers_RNN,
                bidirectional=bidirectional,
                batch_first=True,
                bias=False,
                dropout=self.dropout_rate,
                dtype=dtype
            )

        else:
            self.form_key_layer: nn.Linear = nn.Linear(input_dimension_reflected_to,
                                                       dimension_when_calc_key_halfed * self.bidirectional_int * head_number,
                                                       bias=False,
                                                       dtype=dtype)

        print("Form Key Layer Finish")

        # $$$$$$$$$$$$$$$$$$$$$ For RNN Hidden State $$$$$$$$$$$$$$$$$$$$$
        if self.whether_RNN == "RNN":
            # RNN 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_key_halfed]
            # ，不受batch_first影响
            self.hidden_inout_form_key_layer: torch.Tensor = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                                                                         batch_size,
                                                                         dimension_when_calc_key_halfed * head_number)
        elif self.whether_RNN == "LSTM":
            # LSTM 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_key_halfed]
            # ，不受batch_first影响
            self.hidden_inout_form_key_layer: torch.Tensor = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                                                                         batch_size,
                                                                         dimension_when_calc_key_halfed * head_number)
            self.hidden_LSTM_cell_state: torch.Tensor = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                                                                    batch_size,
                                                                    dimension_when_calc_key_halfed * head_number)
        else:
            pass

        print("Finish Hidden Key Layer Tensor")

        # --------------------- Value ---------------------
        if self.whether_RNN == "RNN":
            # IN [batch_size, input_dimension_reflected_to, seq_len]
            # 随机初始化一个W_{V}矩阵
            # 目的是让网络具有记忆性，因为我过去经验（Attention）和现在的经验是有关联的
            # OUT [batch_size, observer_seq_len, observee_seq_len]
            self.form_value_layer: nn.RNN = nn.RNN(input_size=input_dimension_reflected_to,
                                                   hidden_size=dimension_when_calc_value_halfed * head_number,
                                                   num_layers=self.total_layers_RNN,
                                                   bidirectional=bidirectional,
                                                   batch_first=True,
                                                   bias=False,
                                                   dtype=dtype)
        elif self.whether_RNN == "LSTM":
            # LSTM 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_value_halfed],不受batch_first影响
            self.form_value_layer: nn.LSTM = nn.LSTM(
                input_size=input_dimension_reflected_to,
                hidden_size=dimension_when_calc_value_halfed * head_number,
                num_layers=self.total_layers_RNN,
                bidirectional=bidirectional,
                batch_first=True,
                bias=False,
                dropout=self.dropout_rate,
                dtype=dtype
            )
        else:
            self.form_value_layer: nn.Linear = nn.Linear(input_dimension_reflected_to,
                                                         dimension_when_calc_value_halfed * self.bidirectional_int * head_number,
                                                         bias=False,
                                                         dtype=dtype)

        print("Form Value Layer Finish")

        # $$$$$$$$$$$$$$$$$$$$$ For RNN Hidden State $$$$$$$$$$$$$$$$$$$$$
        if self.whether_RNN == "RNN":
            # RNN 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_value_half]，
            # 不受batch_first影响
            self.hidden_inout_form_value_layer: torch.Tensor \
                = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                              batch_size,
                              dimension_when_calc_value_halfed * head_number)
        elif self.whether_RNN == "LSTM":
            # LSTM 需要一个hidden I/O
            # hidden总是 [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_value_halfed]，
            # 不受batch_first影响
            self.hidden_inout_form_value_layer: torch.Tensor \
                = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                              batch_size,
                              dimension_when_calc_value_halfed * head_number)
            self.hidden_LSTM_cell_state: torch.Tensor \
                = torch.randn(self.total_layers_RNN * self.bidirectional_int,
                              batch_size,
                              dimension_when_calc_value_halfed * head_number)
        else:
            pass

        print("Finish Hidden Value Layer Tensor")
        # ===================== Softmax Layer =====================
        self.observee_attention_score_softmax_layer: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

        # ===================== Normalize Layer =====================
        self.normalize_layer_after_softmax: nn.BatchNorm2d = nn.BatchNorm2d(num_features=sequence_length,
                                                                            eps=1e-5,
                                                                            momentum=0.1,
                                                                            affine=True,
                                                                            track_running_stats=True)

        # ===================== Combine Head =====================
        self.combine_head_result_layer_1: nn.Linear = nn.Linear(in_features=dimension_when_query_halfed
                                                                            * self.bidirectional_int * head_number,
                                                                out_features=dimension_when_query_halfed
                                                                             * self.bidirectional_int * head_number,
                                                                bias=False,
                                                                dtype=dtype)
        self.combine_head_result_layer_2: nn.Linear = nn.Linear(in_features=dimension_when_query_halfed
                                                                            * self.bidirectional_int * head_number,
                                                                out_features=dimension_when_query_halfed * self.bidirectional_int,
                                                                bias=False,
                                                                dtype=dtype)

        # Xavier Initialization
        nn.init.xavier_normal_(self.combine_head_result_layer_1.weight)
        nn.init.xavier_normal_(self.combine_head_result_layer_2.weight)

        # => [batch_size, seq_len, dimension_when_query_halfed * self.bidirectional_int]

        '''
        最后是接受 N 维度的词向量的输入，输出 映射到 N 维度的查询/键/值，并非输出矩阵W_{Q}，W_{K}，W_{V}
        这里的Bi-Directional RNN是为了让网络具有记忆性，因为我过去经验（Attention）和现在的经验是有关联的
        用{Bi-Directional RNN}_{Q}, {Bi-Directional RNN}_{K}, {Bi-Directional RNN}_{V}
        替代了W_{Q}，W_{K}，W_{V}矩阵
        '''

        # ===================== Initialize Self Stored V_{Q}, V_{K}, V_{V} =====================

        # 初始化V_{Q}, V_{K}, V_{V}
        self.__V_Q: torch.Tensor = torch.randn(batch_size,
                                               sequence_length,
                                               head_number,
                                               dimension_when_query_halfed * self.bidirectional_int,
                                               dtype=dtype)
        self.__V_K: torch.Tensor = torch.randn(batch_size,
                                               sequence_length,
                                               head_number,
                                               dimension_when_calc_key_halfed * self.bidirectional_int,
                                               dtype=dtype)
        self.__V_V: torch.Tensor = torch.randn(batch_size,
                                               sequence_length,
                                               head_number,
                                               dimension_when_calc_value_halfed * self.bidirectional_int,
                                               dtype=dtype)

        # ===================== Init Not-Composed-Head Matrix =====================
        self.backup_not_composed_head_matrix: torch.Tensor \
            = torch.randn(batch_size,
                          sequence_length,
                          head_number,
                          dimension_when_query_halfed,
                          dtype=dtype)

        print("Finish ONE-_-- MultiHeadAttention Init")

    def forward(self,
                input_Q: torch.Tensor,
                input_K: torch.Tensor,
                input_V: torch.Tensor
                ) -> torch.Tensor:
        # dimension_when_query_halfed == dimension_when_calc_key_halfed
        # dimension_when_calc_value_diff == input_dimension_reflected_to

        # !===================== Data/Dimension Backup =====================!

        # 暂存 x的维度
        # [batch_size, seq_len, input_dimension_reflected_to]
        batch_size_Q: int = input_Q.size(0)
        seq_len_Q: int = input_Q.size(1)
        # dimension_reflected_to_Q: int = input_Q.size(2)

        # [batch_size, seq_len2, dimension_reflected_to2]
        batch_size_K: int = input_K.size(0)
        seq_len_K: int = input_K.size(1)
        # dimension_reflected_to_K: int = input_K.size(2)

        batch_size_V: int = input_V.size(0)
        seq_len_V: int = input_V.size(1)
        # dimension_reflected_to_V: int = input_V.size(2)

        V_Q: torch.Tensor
        V_K: torch.Tensor
        V_V: torch.Tensor

        # ===================== Query, Key, Value Calculation (Model As Matrix) =====================
        # --------------------- Query ---------------------
        input_Q.cuda()

        # [batch_size, observer_seq_len, observer_dimension_reflected_to]
        # => [batch_size, observer_seq_len, dimension_when_query_halfed * self.bidirectional_int * head_number]
        # hidden [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_query_halfed * head_number]
        if self.whether_RNN == "RNN":
            self.hidden_inout_form_query_layer.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_query_layer = self.hidden_inout_form_query_layer.detach()

            V_Q, self.hidden_inout_form_query_layer = self.form_query_layer(input_Q, self.hidden_inout_form_query_layer)
        elif self.whether_RNN == "LSTM":
            self.hidden_inout_form_query_layer.cuda()
            self.hidden_LSTM_cell_state.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_query_layer = self.hidden_inout_form_query_layer.detach()
            self.hidden_LSTM_cell_state = self.hidden_LSTM_cell_state.detach()

            V_Q, (self.hidden_inout_form_query_layer, self.hidden_LSTM_cell_state) \
                = self.form_query_layer(input_Q,
                                        (self.hidden_inout_form_query_layer, self.hidden_LSTM_cell_state)
                                        )
        else:
            V_Q = self.form_query_layer(input_Q)

        # __V_Q 恢复维度, 因为 dimension_when_query_halfed 是双向的, 所以输出的hidden是两倍的
        V_Q = V_Q.view(batch_size_Q, seq_len_Q, self.head_number,
                       self.dimension_when_query_halfed * self.bidirectional_int)

        # --------------------- Key ---------------------
        input_K.cuda()

        # [batch_size, observee_seq_len, observer_dimension_reflected_to]
        # => [batch_size, observee_seq_len, dimension_when_calc_key_halfed * self.bidirectional_int * head_number]
        # hidden [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_key_halfed * head_number]
        if self.whether_RNN == "RNN":
            self.hidden_inout_form_key_layer.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_key_layer = self.hidden_inout_form_key_layer.detach()

            V_K, self.hidden_inout_form_key_layer = self.form_key_layer(input_K, self.hidden_inout_form_key_layer)
        elif self.whether_RNN == "LSTM":
            self.hidden_inout_form_key_layer.cuda()
            self.hidden_LSTM_cell_state.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_key_layer = self.hidden_inout_form_key_layer.detach()
            self.hidden_LSTM_cell_state = self.hidden_LSTM_cell_state.detach()

            V_K, (self.hidden_inout_form_key_layer, self.hidden_LSTM_cell_state) \
                = self.form_key_layer(input_K,
                                      (self.hidden_inout_form_key_layer, self.hidden_LSTM_cell_state)
                                      )
        else:
            V_K = self.form_key_layer(input_K)

        # __V_K 恢复维度
        V_K = V_K.view(batch_size_K, seq_len_K, self.head_number,
                       self.dimension_when_calc_key_halfed * self.bidirectional_int)

        # --------------------- Value ---------------------
        input_V.cuda()

        # V 为了减少参数，使用两个RNN，一个上一个下 # 这里不使用
        # 这里的RNN/线性层 就是  Input * （W_{V}） = V_{V}
        # 输入 [batch_size, seq_len, input_dimension_reflected_to]
        # 输出 [batch_size, seq_len, dimension_when_calc_value_half * self.bidirectional_int * head_number]
        # HIDDEN [total_layers_RNN * NumberOfDirections, batch_size, dimension_when_calc_value_half]
        if self.whether_RNN == "RNN":
            self.hidden_inout_form_value_layer.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_value_layer = self.hidden_inout_form_value_layer.detach()

            # V_{V} [batch_size, seq_len, number_of_head * dimension_when_calc_value_half]
            V_V, self.hidden_inout_form_value_layer \
                = self.form_value_layer(input_V,
                                        self.hidden_inout_form_value_layer
                                        )
        elif self.whether_RNN == "LSTM":
            self.hidden_inout_form_value_layer.cuda()
            self.hidden_LSTM_cell_state.cuda()

            # 保持隐藏状态，避免梯度回传
            self.hidden_inout_form_value_layer = self.hidden_inout_form_value_layer.detach()
            self.hidden_LSTM_cell_state = self.hidden_LSTM_cell_state.detach()

            V_V, (self.hidden_inout_form_value_layer, self.hidden_LSTM_cell_state) \
                = self.form_value_layer(input_V,
                                        (self.hidden_inout_form_value_layer, self.hidden_LSTM_cell_state)
                                        )
        else:
            # [batch_size, seq_len, input_dimension_reflected_to]
            # => [batch_size, seq_len, number_of_head * dimension_when_calc_value_half * self.bidirectional_int]
            V_V = self.form_value_layer(input_V)

        # change Nan to 0
        inf_mask = torch.isinf(V_V)
        nan_mask = torch.isnan(V_V)
        # combine
        mask = inf_mask | nan_mask
        V_V[mask] = 1
        # @Debug Check if there is  Nan
        Toolbox.test_nan(V_V)

        # __V_V 恢复维度
        # => [batch_size, seq_len, number_of_head, dimension_when_calc_value_half]
        V_V = V_V.view(batch_size_V, seq_len_V, self.head_number,
                       self.dimension_when_calc_value_half * self.bidirectional_int)

        # --------------------- Release Memory ---------------------
        # del input_Q
        del input_K
        del input_V
        torch.cuda.empty_cache()
        gc.collect()

        # --------------------- Self Q K V Update ---------------------

        # 更新V_{Q}, V_{K}, V_{V}
        self.__V_Q = V_Q
        self.__V_K = V_K
        self.__V_V = V_V

        # print(self.__V_Q.shape, self.__V_K.shape, self.__V_V.shape)

        # --------------------- Head Transpose ---------------------

        # 把 head 放到 seq_len 前面
        # [batch_size, seq_len, number_of_head, dimension_when_query_halfed]
        # => [batch_size, number_of_head, seq_len, dimension_when_query_halfed]
        V_Q = V_Q.transpose(1, 2)
        V_K = V_K.transpose(1, 2)
        V_V = V_V.transpose(1, 2)

        # ===================== Attention Calculation =====================
        # 对于每个词，和另外一个词， Query 和 Key 相乘 就是 Attention.
        # Attention = softmax(Query * Key) * Value
        # [batch_size, head, observer_seq_len, dimension_when_query_halfed]
        # * [batch_size, head, observee_seq_len, dimension_when_calc_key_halfed]
        # ASSUME dimension_when_query_halfed == dimension_when_calc_key_halfed
        # THEREFORE => [batch_size, number_of_head, observer_seq_len, observee_seq_len]
        attention_score: torch.Tensor = torch.matmul(V_Q, V_K.transpose(-2, -1))

        if not self.mask_bool:
            # 手动释放显存
            del V_Q
            del V_K
            torch.cuda.empty_cache()
            gc.collect()

        # 为了稳定性，除以 dimension_when_query_halfed 或者 dimension_when_calc_key_halfed 的平方根
        # 这里为了兼容性，使用二者相乘之后的平方根
        # 为了使得梯度更加稳定，避免梯度爆炸
        attention_score = attention_score / numpy.sqrt(
            numpy.sqrt(self.dimension_when_query_halfed
                       * self.dimension_when_calc_key_halfed)
        )

        # Check if there is  Nan
        Toolbox.test_nan(attention_score)

        # ===================== Masking =====================
        if self.mask_bool:
            # 和3Blue1Brown的视频不同，这里的Q是纵向的，K是横向的
            # 先Mask
            # 使得在计算Attention的时候，不会考虑到padding的词
            #  由于V_Q 和 __V_K 的维度是相同的，所以这里的mask 直接对角线 上半部分遮蔽，True，下半部分不遮蔽，False
            mask_bool: torch.Tensor = Toolbox.create_mask(V_Q, V_K)

            mask_bool: torch.Tensor = mask_bool.to(self.device)

            # Mask 操作
            # mask_fill 返回一个新的张量，mask_fill_则是直接在原来的张量上进行操作
            attention_score = attention_score.masked_fill(mask_bool, -1e9)

        # ===================== Softmax =====================

        # softmax 归一化, along the first seq_len dimension
        # 归一化第最后一个维度(-1)，也就是每个 Q 的所有 K
        # [batch_size, number_of_head, observer_seq_len, observee_seq_len]
        attention_score = self.observee_attention_score_softmax_layer(attention_score)

        '''
         Adaptive Attention Span 之后学习，思想是除去Attention 的无效计算，因为每个头专注的范围是不同的
        '''

        # Final adjust_rate
        # torch.matmul 两个矩阵相乘
        # 三维矩阵相乘 [batch_size, seq_len1, input_dimension_reflected_to]
        # 会自动广播，不管batch_size

        # ===================== Attention Calculation =====================

        # [batch_size, number_of_head, observer_seq_len, observee_seq_len]
        # * [batch_size, number_of_head, observee_seq_len, dimension_when_calc_value_half]
        # => [batch_size, number_of_head, observer_seq_len, dimension_when_calc_value_half]
        rated_adjust_vector: torch.Tensor = torch.matmul(attention_score, V_V)

        # ===================== Adjust Original Vector by Attention =====================

        output: torch.Tensor = rated_adjust_vector

        # Check if there is  Nan
        Toolbox.test_nan(rated_adjust_vector)

        # 残差连接, 使得网络更加稳定
        output = output + input_Q.unsqueeze(1)

        # ===================== Combine Head =====================
        # --------------------- Preparation & Backup ---------------------
        # [batch_size, number_of_head, observer_seq_len, dimension_when_query_halfed]
        # => [batch_size, observer_seq_len, number_of_head, dimension_when_query_halfed]
        # Change to Continuous Matrix & Transpose & Backup
        output = output.transpose(1, 2).contiguous()

        # Backup: [batch_size, seq_len, number_of_head, dimension_when_query_halfed * self.bidirectional_int]
        # So, Here Suits
        self.backup_not_composed_head_matrix = output

        # @Debug Check if there is  Nan
        Toolbox.test_nan(output)

        # => [batch_size * observer_seq_len, number_of_head * dimension_when_query_halfed * self.bidirectional_int]
        output = output.view(batch_size_Q * seq_len_Q,
                             self.head_number * self.dimension_when_query_halfed * self.bidirectional_int)

        # change Nan to 0
        inf_mask = torch.isinf(output)
        nan_mask = torch.isnan(output)
        # combine
        mask = inf_mask | nan_mask
        output[mask] = 1
        # @Debug Check if there is  Nan
        Toolbox.test_nan(output)

        output = torch.nn.functional.normalize(input=output, p=2, dim=-1)
        original_output: torch.Tensor = output
        # --------------------- Final Combination ---------------------
        # through the layer & combination
        # @ChangeShape [batch_size * observer_seq_len, number_of_head * dimension_when_query_halfed * self.bidirectional_int]
        # => [batch_size * observer_seq_len, number_of_head * dimension_when_query_halfed * self.bidirectional_int]
        output = self.combine_head_result_layer_1(output)

        # 残差连接, 使得网络更加稳定
        output = output + original_output

        # change Nan to 0
        inf_mask = torch.isinf(output)
        nan_mask = torch.isnan(output)
        # combine
        mask = inf_mask | nan_mask
        output[mask] = 1

        output = torch.nn.functional.normalize(input=output, p=2, dim=-1)

        # @Debug Check if there is  Nan
        Toolbox.test_nan(output)
        # NonLinear Activation
        output = torch.selu(output)
        # @Debug Check if there is  Nan
        Toolbox.test_nan(output)

        # @ChangeShape [batch_size * observer_seq_len, number_of_head * dimension_when_query_halfed * self.bidirectional_int]
        # => [batch_size, observer_seq_len, dimension_when_query_halfed * self.bidirectional_int]
        output = self.combine_head_result_layer_2(output)

        # change Nan to 0
        inf_mask = torch.isinf(output)
        nan_mask = torch.isnan(output)
        # combine
        mask = inf_mask | nan_mask
        output[mask] = 1

        output = torch.nn.functional.normalize(input=output, p=2, dim=-1)

        # [batch_size, observer_seq_len, dimension_when_query_halfed * self.bidirectional_int]
        output = output + input_Q

        # @Debug Check if there is  Nan
        Toolbox.test_nan(output)

        # @ChangeShape [batch_size * observer_seq_len, dimension_when_query_halfed * self.bidirectional_int]
        # => [batch_size, observer_seq_len, dimension_when_query_halfed * self.bidirectional_int]
        output = output.view(batch_size_Q, seq_len_Q, self.dimension_when_query_halfed * self.bidirectional_int)

        # Since input_Q does not have HEAD,
        # Add the head to compatible with rated_adjust_vector
        # [batch_size, observer_seq_len, dimension_when_query_halfed]
        # => [batch_size, number_of_head, observer_seq_len, dimension_when_query_halfed]
        # # original_observer_Q: torch.Tensor = input_Q.unsqueeze(1)
        # 修正高维映射的向量
        # rated_adjust_vector [batch_size, number_of_head, observer_seq_len, dimension_when_calc_value_half]
        # input_Q [batch_size, number_of_head, observer_seq_len, dimension_when_query_halfed]
        # dimension_when_calc_value_half == dimension_when_query_halfed
        # output: torch.Tensor = original_observer_Q + output

        return output

    def get_Q_matrix(self) -> torch.Tensor:
        return self.__V_Q

    def get_K_matrix(self) -> torch.Tensor:
        return self.__V_K

    def get_V_matrix(self) -> torch.Tensor:
        return self.__V_V


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
                                  device=self.device,
                                  dtype=self.data_type,)
        self.lstm_key = nn.LSTM(num_layers=number_of_layers,
                                input_size=feature_size,
                                hidden_size=lstm_hidden_dim,
                                device=self.device,
                                  dtype=self.data_type,)
        self.lstm_value = nn.LSTM(num_layers=number_of_layers,
                                  input_size=feature_size,
                                  hidden_size=lstm_hidden_dim,
                                  device=self.device,
                                  dtype=self.data_type,)

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
            add_bias_kv=bias,
            bias=bias,
            dtype=dtype,
            device=device
        )

    def reinit_hidden_cell_state(self):
        # 手动释放显存
        del self.lstm_query_cell_state
        del self.lstm_key_cell_state
        del self.lstm_value_cell_state
        del self.lstm_query_hidden_state
        del self.lstm_key_hidden_state
        del self.lstm_value_hidden_state
        gc.collect()
        torch.cuda.empty_cache()



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

    @memory_profiler.profile()
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor
                ) -> Tuple[Tensor]:
        # 使用LSTM层计算Q, K, V

        # 解耦
        # self.lstm_query_hidden_state = self.lstm_query_hidden_state.detach()
        # self.lstm_query_cell_state = self.lstm_query_cell_state.detach()
        # self.lstm_key_hidden_state = self.lstm_key_hidden_state.detach()
        # self.lstm_key_cell_state = self.lstm_key_cell_state.detach()
        # self.lstm_value_hidden_state = self.lstm_value_hidden_state.detach()
        # self.lstm_value_cell_state = self.lstm_value_cell_state.detach()

        query_output: typing.Optional[Tensor] = None
        key_output: typing.Optional[Tensor] = None
        value_output: typing.Optional[Tensor] = None

        # [seq_len, batch, hidden_size]
        query_output, (lstm_query_hidden_state, lstm_query_cell_state) \
            = self.lstm_query(query,
                              (self.lstm_query_hidden_state,
                               self.lstm_query_cell_state))
        self.lstm_query_cell_state = lstm_query_cell_state
        self.lstm_query_hidden_state = lstm_query_hidden_state

        key_output, (lstm_key_hidden_state, lstm_key_cell_state) \
            = self.lstm_key(key,
                            (self.lstm_key_hidden_state,
                             self.lstm_key_cell_state))
        self.lstm_key_cell_state = lstm_key_cell_state
        self.lstm_key_hidden_state = lstm_key_hidden_state



        value_output, (lstm_value_hidden_state, lstm_value_cell_state) \
            = self.lstm_value(value,
                              (self.lstm_value_hidden_state,
                               self.lstm_value_cell_state))
        self.lstm_value_cell_state = lstm_value_cell_state
        self.lstm_value_hidden_state = lstm_value_hidden_state

        # Multihead Attention
        # [seq_len, batch, embed_dim]
        # => [seq_len, batch, embed_dim]
        output, attention = self.multihead_attention(query_output, key_output, value_output)

        query: typing.Optional[torch.Tensor] = None
        key: typing.Optional[torch.Tensor] = None
        value: typing.Optional[torch.Tensor] = None
        query_output: typing.Optional[torch.Tensor] = None
        key_output: typing.Optional[torch.Tensor] = None
        value_output: typing.Optional[torch.Tensor] = None
        attention: typing.Optional[torch.Tensor] = None
        del attention
        del query, query_output
        del key, key_output
        del value, value_output
        gc.collect()
        torch.cuda.empty_cache()

        # @output: [seq_len, batch, embed_dim]
        return output
