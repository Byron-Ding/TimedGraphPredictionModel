import torch
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.PositionalEncodingLayers import PositionalEncoding_AllFeatures
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.CNNFeatureExtractionNetwork import CNNFeatureExtractionNetwork
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.MultiHeadAttention import LSTMMultiheadAttention
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.PoolingLayer import PoolingLayer
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.FinalFullConnectionLayer import FinalFullConnectionLayer
from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.PositionalEncodingLayers import PositionalEncoding_TimeStamp

from typing import Final

# from memory_profiler import profile


class ImprovedModelWithTimeSlot(torch.nn.Module):
    """
    @Input: [Batch_size, Time_slot, Feature_number, graph_height, graph_width]

    """

    def __init__(self,
                 batch_size: int,
                 max_time_slot: int,
                 feature_number: int,
                 graph_height: int,
                 graph_width: int,
                 cnn_conv_stride: int,
                 attention_head_number: int,
                 lstm_time_attention_layer_number: int,
                 final_linear_layer_hidden_size: int,
                 data_type: torch.dtype,
                 device: torch.device,
                 ):
        """

        :param batch_size:
        :param max_time_slot:
        :param feature_number:
        :param graph_height:
        :param graph_width:
        :param cnn_conv_stride:
        :param attention_head_number:
        :param lstm_time_attention_layer_number:
        :param final_linear_layer_hidden_size:
        :param data_type:
        :param device:
        """
        super(ImprovedModelWithTimeSlot, self).__init__()

        # ================================== Parameter Backup ==================================
        # Matrix Parameter
        self.batch_size: int = batch_size
        self.max_time_slot: int = max_time_slot
        self.feature_number: int = feature_number
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width

        # model behavior parameter
        self.cnn_conv_stride: int = cnn_conv_stride
        self.attention_head_number: int = attention_head_number

        # Model structure parameter
        self.cnn_conv_extracted_number: Final[int] = 7
        self.lstm_time_attention_layer_number: int = lstm_time_attention_layer_number

        # Hardware Parameter
        self.data_type: torch.dtype = data_type
        self.device: torch.device = device

        print("@Init::ImprovedModelWithTimeSlot::Finish Parameter Backup")
        # ================================== Data Normalization ==================================
        # 防止梯度消失，数据归一化，因为不同层的数据可能有的100000 有的只有 1
        # @InputRequired: [Batch, Channel, Height, Width]
        # @Output: [Batch, Channel, Height, Width]
        self.enter_batch_normalization: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            dtype=data_type,
            device=device,
        )

        # ================================== 分割，一个一个来 ==================================
        # 首先还是 Positional Encoding，混入图片像素坐标信息，特征标签，卫星地面夹角标签
        # @InputRequired: [batch_size, feature_number, graph_height, graph_width]
        # @Output: [batch_size, light_layer, graph_height, graph_width]
        self.positional_encoding_all_features: PositionalEncoding_AllFeatures = PositionalEncoding_AllFeatures(
            feature_number=feature_number, graph_height=graph_height, graph_width=graph_width, dtype=data_type,
            device=device)
        print("@Init::ImprovedModelWithTimeSlot::Finish Positional Encoding Init")

        # CNN Layer，进入CNN 提取特征
        # @InputRequired: [batch_size, feature_number, graph_height, graph_width]
        # @Output: [batch_size, feature_number, graph_height, graph_width, extracted_graph_feature]
        self.cnn_feature_extraction_network: CNNFeatureExtractionNetwork = CNNFeatureExtractionNetwork(
            batch_size=batch_size,
            feature_number=feature_number,
            height=graph_height,
            width=graph_width,
            cnn_conv_stride=cnn_conv_stride,
            device=device,
            dtype=data_type
        )
        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish CNN Feature Extraction Network Init "
              "")

        # 输出多了一个维度，特征组，需要改变类型
        # @ChangeShape: [batch_size, feature_number, graph_height, graph_width, extracted_graph_feature]
        # => [batch_size, feature_number, extracted_graph_feature, graph_height, graph_width]
        # => [batch_size, feature_number * extracted_graph_feature, graph_height, graph_width]

        # ================================== Pooling Layer ==================================
        # 不pooling 太大了，数据太多了
        # @InputRequired: [batch_size, feature_number, height, width]
        # @Output: [batch_size, feature_number, height/32, width/32]
        self.pooling_layer = PoolingLayer(batch_size=batch_size, feature_number=feature_number,
                                          graph_height=graph_height, graph_width=graph_width, device=device,
                                          dtype=data_type)

        self.pooling_divide: Final[int] = self.pooling_layer.pooling_divide
        self.new_graph_height = graph_height // self.pooling_divide
        self.new_graph_width = graph_width // self.pooling_divide

        print("The divided graph height is: ", self.new_graph_height,
              "The divided graph width is: ", self.new_graph_width)

        # Change the shape of the output
        # [batch_size, feature_number, graph_height/32, graph_width/32]

        print("@Init::ImprovedModelWithTimeSlot::Finish Pooling Layer Init")

        # 备份

        # ================================== Multihead Attention Layer ==================================
        # 这里的 Multihead Attention Layer 是为了学习不同像素也就是不同位置之间的关系，
        # 卷积特征以及观测维度 本质都是维度，所以需要放到维度层
        # 因为我们是按照时间序列输入的，所以上次注意的位置可能之后会漂移，
        # 比如我现在看到有个台风在左下角，下一张照片可能就跑到中间来了
        # @InputRequired: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
        # @Output: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
        self.lstmMultiheadAttention: LSTMMultiheadAttention \
            = LSTMMultiheadAttention(batch_size=batch_size,
                                     feature_size=feature_number * self.cnn_conv_extracted_number,
                                     num_heads=attention_head_number,
                                     number_of_layers=lstm_time_attention_layer_number,
                                     lstm_hidden_dim=feature_number * self.cnn_conv_extracted_number,
                                     dropout=0, bias=True,
                                     dtype=data_type, device=device)

        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish Multihead Attention Layer Init "
              "")

        # 残差
        # 残差之后要Normalization

        # ================================== Normalization Layer ==================================
        # 因为我们不知道Attention 之后 Layer 会被加成什么样，这个输出的WV是不确定的。
        # 有可能会有非常大的值，也有可能有非常小的值，所以我们需要一个Normalization Layer
        # @InputRequired: [batch_size, new_graph_height * new_graph_width,
        # feature_number * extracted_graph_feature]
        # @Output: [batch_size, new_graph_height * new_graph_width, feature_number * extracted_graph_feature]
        self.after_multihead_attention_batch_normalization: torch.nn.LayerNorm = torch.nn.LayerNorm(
            normalized_shape=[feature_number * self.cnn_conv_extracted_number],
            eps=1e-05,  # Default Value - 加入一个很小的数，防止分母为0
            elementwise_affine=True,  # Default Value - 是否加入可学习的参数
            bias=True,  # Default Value - 是否加入偏置 正常情况下是True，防止两边分布一模一样
            dtype=data_type,
            device=device
        )

        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish Normalization Layer Init "
              "")

        # ================================== Time Stamp Positional Encoding ==================================
        # 时间戳的位置编码
        # 进入时间相关的层之前提供
        # @InputRequired: Any Shape
        # @Output: Same Shape as Input
        # @InputActual: [batch_size,
        #                new_graph_height * new_graph_width,
        #                feature_number * extracted_graph_feature]
        self.positional_encoding_time_stamp: PositionalEncoding_TimeStamp = PositionalEncoding_TimeStamp()

        # @ChangeShape: [batch_size,
        # new_graph_height * new_graph_width,
        # feature_number * extracted_graph_feature]
        # => [new_graph_height * new_graph_width, batch_size, feature_number * extracted_graph_feature]

        # ================================== LSTM Relation Output Layer ==================================
        # LSTM Layer
        # @InputRequired: [sequence_length, batch_size, dimension_reflected_to + dimension_reflected_to]
        # @Output: [sequence_length, batch_size, dimension_reflected_to]
        # @InputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
        self.lstm_time_attention_layer = torch.nn.LSTM(input_size=feature_number * self.cnn_conv_extracted_number * 2,
                                                       hidden_size=feature_number * self.cnn_conv_extracted_number,
                                                       num_layers=lstm_time_attention_layer_number
                                                       )

        # hidden cell state
        self.lstm_time_attention_layer_hidden_state: torch.Tensor = torch.randn(
            size=(lstm_time_attention_layer_number,
                  batch_size,
                  feature_number * self.cnn_conv_extracted_number),
            dtype=data_type,
            device=device
        )
        self.lstm_time_attention_layer_cell_state: torch.Tensor = torch.randn(
            size=(lstm_time_attention_layer_number,
                  batch_size,
                  feature_number * self.cnn_conv_extracted_number),
            dtype=data_type,
            device=device
        )

        self.after_lstm_attention_relu: torch.nn.ReLU = torch.nn.ReLU()

        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish LSTM Relation Output Layer Init "
              "")

        # ================================== Time attention Normalization Layer ==================================
        self.after_time_attention_batch_normalization: torch.nn.LayerNorm = torch.nn.LayerNorm(
            normalized_shape=[feature_number * self.cnn_conv_extracted_number],
            eps=1e-05,  # Default Value - 加入一个很小的数，防止分母为0
            elementwise_affine=True,  # Default Value - 是否加入可学习的参数
            bias=True,  # Default Value - 是否加入偏置 正常情况下是True，防止两边分布一模一样
            dtype=data_type,
            device=device
        )

        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish Time attention Normalization Layer Init "
              "")

        # ================================== Time Difference Sequence LSTM Layer ==================================
        # LSTM Layer
        # 这个LSTM Layer 是为了学习时间序列之间的关系，但是记住当前的时间是啥，不能回去就忘记了我们现在是在哪个时间点
        # @InputRequired: [sequence_length, batch_size, dimension_reflected_to]
        # @Output: [sequence_length, batch_size, dimension_reflected_to]

        self.tanh_reflected_to: torch.nn.Tanh = torch.nn.Tanh()

        self.lstm_time_recall_but_target_from_current_layer: torch.nn.LSTM = \
            torch.nn.LSTM(input_size=feature_number * self.cnn_conv_extracted_number,
                          hidden_size=feature_number * self.cnn_conv_extracted_number,
                          num_layers=lstm_time_attention_layer_number
                          )

        # hidden cell state
        self.lstm_time_recall_but_target_from_current_layer_hidden_state: torch.Tensor = torch.randn(
            size=(lstm_time_attention_layer_number,
                  batch_size,
                  feature_number * self.cnn_conv_extracted_number),
            dtype=data_type,
            device=device
        )
        self.lstm_time_recall_but_target_from_current_layer_cell_state: torch.Tensor = torch.randn(
            size=(lstm_time_attention_layer_number,
                  batch_size,
                  feature_number * self.cnn_conv_extracted_number),
            dtype=data_type,
            device=device
        )



        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish Time Difference Sequence LSTM Layer Init "
              "")

        # ================================== Final Fully Connected Layer ==================================
        # 最后的全连接层
        # @InputRequired: [batch_size, input_size]
        # @Output: [batch_size, output_size]
        self.final_full_connection_layer = FinalFullConnectionLayer(
            batch_size=batch_size,
            input_size=
            self.new_graph_height * self.new_graph_width
            * feature_number * self.cnn_conv_extracted_number,
            hidden_size=final_linear_layer_hidden_size,
            output_size=self.graph_height * self.graph_width,
            device=device,
            dtype=data_type
        )

        self.current_time_slot_index: int = 0

        self.current_time_slot_pooled_data_SBD: torch.Tensor = torch.randn(
            size=(feature_number * self.cnn_conv_extracted_number,
                  batch_size,
                  self.new_graph_height * self.new_graph_width),
            dtype=data_type,
            device=device
        )
        self.current_time_slot_pooled_data_SBD_cumulated: torch.Tensor = torch.randn(
            size=(feature_number * self.cnn_conv_extracted_number,
                  batch_size,
                  self.new_graph_height * self.new_graph_width),
            dtype=data_type,
            device=device
        )
        self.current_time_slot_pooled_data_updated_flag: bool = False

        print(""
              "@Init::ImprovedModelWithTimeSlot::Finish Final Fully Connected Layer Init "
              "")

        print("================= @Init::ImprovedModelWithTimeSlot::FINISH INIT =================")

    def reinit_lstm_time_attention_layer_hidden_state(self) -> None:
        self.lstm_time_attention_layer_hidden_state = torch.randn(
            size=(self.lstm_time_attention_layer_number,
                  self.batch_size,
                  self.feature_number * self.cnn_conv_extracted_number),
            dtype=self.data_type,
            device=self.device
        )
        self.lstm_time_attention_layer_cell_state = torch.randn(
            size=(self.lstm_time_attention_layer_number,
                  self.batch_size,
                  self.feature_number * self.cnn_conv_extracted_number),
            dtype=self.data_type,
            device=self.device
        )

    def reinit_lstm_time_recall_but_target_from_current_layer_hidden_state(self) -> None:
        self.lstm_time_recall_but_target_from_current_layer_hidden_state = torch.randn(
            size=(self.lstm_time_attention_layer_number,
                  self.batch_size,
                  self.feature_number * self.cnn_conv_extracted_number),
            dtype=self.data_type,
            device=self.device
        )
        self.lstm_time_recall_but_target_from_current_layer_cell_state = torch.randn(
            size=(self.lstm_time_attention_layer_number,
                  self.batch_size,
                  self.feature_number * self.cnn_conv_extracted_number),
            dtype=self.data_type,
            device=self.device
        )

    def forward(self,
                time_series_layer_data: torch.Tensor,
                satellite_ground_cosine_graph: torch.Tensor,
                light_wavelength: torch.Tensor,
                time_label: torch.Tensor) -> torch.Tensor:
        """
        @InputRequired: [Batch_size, Time_slot, Feature_number, graph_height, graph_width]
        @OutputAvailable: [Batch_size, graph_height, graph_width]

        :param time_series_layer_data:
        :param satellite_ground_cosine_graph:
        :param light_wavelength:
        :param time_label:
        :return:
        """
        # 把time_label 挪到GPU上
        time_label: torch.Tensor = time_label.to(dtype=self.data_type, device=self.device)

        time_length: int = time_series_layer_data.size(dim=1)

        # 检查是否循环过度，超过最大时间长度
        if time_length > self.max_time_slot:
            raise ValueError(f"time_length[{time_length}] is larger than max_time_slot[{self.max_time_slot}]")

        for current_time_slot_index in range(time_length):
            print("@forward::ImprovedModelWithTimeSlot::Entering the Time Slot: ", current_time_slot_index)

            # [Batch_size, Time_slot, Feature_number, graph_height, graph_width]
            # 先取出当前的时间点的切片
            # [Batch_size, Feature_number, graph_height, graph_width]
            current_time_data: torch.Tensor = time_series_layer_data[:, current_time_slot_index, :, :, :]

            # Batch Normalization
            # @Input: [B, C, H, W]
            # @Output: [B, C, H, W]
            # @InputActual: [Batch_size, Feature_number, graph_height, graph_width]
            # @OutputActual: [Batch_size, Feature_number, graph_height, graph_width]
            current_time_data = self.enter_batch_normalization(current_time_data)

            with torch.no_grad():
                # Positional Encoding
                # @Input: [batch_size, feature_number, graph_height, graph_width]
                # @Output: [batch_size, feature_number, graph_height, graph_width]
                # @InputActual: [Batch_size, Feature_number, graph_height, graph_width]
                # @OutputActual: [Batch_size, Feature_number, graph_height, graph_width]
                current_time_data = self.positional_encoding_all_features(
                    x=current_time_data,
                    satellite_ground_cosine_graph=satellite_ground_cosine_graph,
                    light_wavelength=light_wavelength
                )

                print("@forward::ImprovedModelWithTimeSlot::Finish Positional Encoding")

            # CNN Layer
            # @Input: [batch_size, feature_number, graph_height, graph_width]
            # @Output: [batch_size, feature_number, graph_height, graph_width, extracted_graph_feature]
            # @InputActual: [Batch_size, Feature_number, graph_height, graph_width]
            # @OutputActual: [Batch_size, Feature_number, graph_height, graph_width, extracted_graph_feature]
            current_time_data = self.cnn_feature_extraction_network(current_time_data)

            print("@forward::ImprovedModelWithTimeSlot::Finish CNN Feature Extraction Network")

            # Change the shape of the output
            # @ChangeShape: [batch_size, feature_number, graph_height, graph_width, extracted_graph_feature]
            # => [batch_size, graph_height, graph_width, feature_number, extracted_graph_feature]
            current_time_data_ = current_time_data.permute(0, 2, 3, 1, 4)

            # => [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            current_time_data = current_time_data_.resize(self.batch_size,
                                                          self.graph_height * self.graph_width,
                                                          self.feature_number * self.cnn_conv_extracted_number)

            # @ChangeShape: [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            # => [batch_size, feature_number * extracted_graph_feature, graph_height * graph_width]
            current_time_data_ = current_time_data.permute(0, 2, 1)

            # @ChangeShape: [batch_size, feature_number * extracted_graph_feature, graph_height * graph_width]
            # => [batch_size, feature_number * extracted_graph_feature, graph_height, graph_width]
            current_time_data = current_time_data_.view(self.batch_size,
                                                        self.feature_number * self.cnn_conv_extracted_number,
                                                        self.graph_height,
                                                        self.graph_width)

            # Pooling Layer
            # @Input: [batch_size, feature_number, height, width]
            # @Output: [batch_size, feature_number, height/32, width/32]
            # @InputActual: [batch_size, feature_number * extracted_graph_feature, graph_height, graph_width]
            # @OutputActual: [batch_size,
            # feature_number * extracted_graph_feature,
            # graph_height/32, graph_width/32]
            current_time_data = self.pooling_layer(current_time_data)

            print("@forward::ImprovedModelWithTimeSlot::Finish Pooling Layer")

            # @ChangeShape: [batch_size,
            # feature_number * extracted_graph_feature,
            # graph_height/32, graph_width/32]
            # => [batch_size, feature_number * extracted_graph_feature, graph_height/32 * graph_width/32]
            current_time_data = current_time_data.view(self.batch_size,
                                                       self.feature_number * self.cnn_conv_extracted_number,
                                                       self.new_graph_height * self.new_graph_width)
            # => [graph_height/32 * graph_width/32, feature_number * extracted_graph_feature, batch_size]
            current_time_data = current_time_data.permute(2, 0, 1)

            # copy it as the residual connection later 残差
            current_time_data_copy_after_cnn_SBD: torch.Tensor = current_time_data

            print("@forward::ImprovedModelWithTimeSlot::Finish Copying Data & Residual Connection")



            # Multihead Attention Layer
            # @Input: [seq_len, batch_size, embed_dim] \n
            # @Output: [seq_len, batch_size, embed_dim]
            # @InputActual: [graph_height/N * graph_width/N,
            # batch_size,
            # feature_number * extracted_graph_feature]
            # @OutputActual: [graph_height/N * graph_width/N,
            # batch_size,
            # feature_number * extracted_graph_feature]
            current_time_data, attention = self.lstmMultiheadAttention(current_time_data,
                                                                       current_time_data,
                                                                       current_time_data)

            print("@forward::ImprovedModelWithTimeSlot::Finish Multihead Attention Layer")

            # Residual Connection
            # @Input1: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @Input2: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @Output: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            current_time_data = current_time_data + current_time_data_copy_after_cnn_SBD

            # @ChangeShape: [graph_height/N * graph_width/N,
            # batch_size,
            # feature_number * extracted_graph_feature]
            # => [batch_size, graph_height/N * graph_width/N, feature_number * extracted_graph_feature]
            current_time_data = current_time_data.permute(1, 0, 2)

            # ================================== Normalization Layer ==================================
            # Normalization Layer
            # @Input: [batch_size, graph_height/N * graph_width/N, feature_number * extracted_graph_feature]
            # @Output: [batch_size, graph_height/N * graph_width/N, feature_number * extracted_graph_feature]
            current_time_data = self.after_multihead_attention_batch_normalization(current_time_data)

            print("@forward::ImprovedModelWithTimeSlot::Finish Normalization Layer")

            # ================================== Time Stamp Positional Encoding ==================================
            # time_label: [batch_size, time_slot]
            # Extend the time label to the same shape as the current_time_data
            # @Input: [batch_size, time_slot]
            # @Output: [batch_size, time_slot, 1]
            # 只有一个数字的维度 比如 [a, b, 1] 的那个1， 在切片的时候会自动被舍去，
            # 如果要广播维度做加法，就需要这个1，所以要先切片，再扩展维度unsqueeze
            with torch.no_grad():
                time_label_sliced: torch.Tensor = time_label[:, current_time_slot_index]
                # 去值之后变成了一维数组
                time_label_sliced = time_label_sliced.unsqueeze(dim=1).unsqueeze(dim=2)
                current_time_data = self.positional_encoding_time_stamp(current_time_data,
                                                                        time_label_sliced)

                print("@forward::ImprovedModelWithTimeSlot::Finish Time Stamp Positional Encoding")

            # ================================== LSTM Relation Output Layer ==================================
            # Change the shape of the output
            # @ChangeShape: [batch_size,
            # graph_height/N * graph_width/N,
            # feature_number * extracted_graph_feature]
            # => [batch_size, feature_number * extracted_graph_feature, graph_height/N * graph_width/N]
            current_time_data = current_time_data.permute(0, 2, 1)
            # => [batch_size, feature_number * extracted_graph_feature, graph_height/N * graph_width/N]
            current_time_data = current_time_data.view(self.batch_size,
                                                       self.feature_number * self.cnn_conv_extracted_number,
                                                       self.new_graph_height,
                                                       self.new_graph_width)

            # [batch_size, feature_number * extracted_graph_feature, graph_height/N, graph_width/N]
            # @ChangeShape: [batch_size,
            # feature_number * extracted_graph_feature,
            # graph_height/N, graph_width/N]
            # => [batch_size, feature_number * extracted_graph_feature, graph_height/N * graph_width/N]
            current_time_data = current_time_data.view(self.batch_size,
                                                       self.feature_number * self.cnn_conv_extracted_number,
                                                       self.new_graph_height * self.new_graph_width)

            # => [sequence_length, batch_size, dimension_reflected_to + dimension_reflected_to]
            # => [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            current_time_data = current_time_data.permute(2, 0, 1)

            # 变换完成，进行备份，以备后续使用，以及残差，后面这个变量名会被覆盖
            if current_time_slot_index == 0:
                self.current_time_slot_pooled_data_SBD = current_time_data
                self.current_time_slot_pooled_data_SBD_cumulated = current_time_data
                self.current_time_slot_pooled_data_updated_flag = True

                # 第一个就和自己拼接，是百分百的相关性

            current_time_data_copy_after_cross_time_compare_BSD: torch.Tensor = current_time_data

            # 首先测试是否已经备份，否则报错
            if not self.current_time_slot_pooled_data_updated_flag:
                raise AssertionError("The current time slot data is not updated, please check the code")

            # 真正的拼接数据
            # @Output: [sequence_length, batch_size, dimension_reflected_to + dimension_reflected_to]
            current_time_data = torch.cat((self.current_time_slot_pooled_data_SBD, current_time_data), dim=2)

            print("@forward::ImprovedModelWithTimeSlot::Finish Concatenation Data Together")

            # LSTM Layer 判断时间过去是否值得注意
            # @InputRequired: [sequence_length, batch_size, dimension_reflected_to + dimension_reflected_to]
            # @Output: [sequence_length, batch_size, dimension_reflected_to]
            # @InputActual: [graph_height * graph_width,
            # batch_size,
            # feature_number * extracted_graph_feature * 2]
            # @OutputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]

            # 解耦
            self.lstm_time_attention_layer_hidden_state = self.lstm_time_attention_layer_hidden_state.detach()
            self.lstm_time_attention_layer_cell_state = self.lstm_time_attention_layer_cell_state.detach()

            current_time_data, (self.lstm_time_attention_layer_hidden_state,
                                self.lstm_time_attention_layer_cell_state) = \
                self.lstm_time_attention_layer(current_time_data,
                                               (self.lstm_time_attention_layer_hidden_state,
                                                self.lstm_time_attention_layer_cell_state)
                                               )

            print("@forward::ImprovedModelWithTimeSlot::Finish LSTM Relation Output Layer")

            # LSTM 输出的值域在 -1 到 1 之间，0代表没有关系，1代表有关系，-1代表负相关，已经不需要注意了，过头了
            # 看一下最大值是否超过0.01，如果超过0.01，就说明有关系，否则就没有关系
            # 如果没有关系，就不需要再往后看了
            # @Input: [sequence_length, batch_size, dimension_reflected_to]
            # @Output: [sequence_length, batch_size, dimension_reflected_to]
            # @InputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @OutputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # 不填dimension，就是全局最大值，返回size为1的矩阵
            current_time_data_max_value: torch.Tensor = torch.max(current_time_data)
            # #$$ print(current_time_data_max_value.shape)
            current_time_data_max_value: int = current_time_data_max_value.item()

            # if negative related, give it up, 舍弃负相关的数据
            current_time_data = self.after_lstm_attention_relu(current_time_data)

            # 每个像素的相关性，乘以过去的那个时间点的数据，得到相关性矩阵
            # @Input1: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @Input2: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @Output: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # 点乘，每个元素乘以对应的元素
            current_time_data = torch.mul(current_time_data, current_time_data_copy_after_cross_time_compare_BSD)

            with torch.no_grad():
                # + A very small number, to prevent the output is 0. Then we can't get the gradient
                current_time_data = current_time_data + 1e-5

            # + Current, let the current time know the past
            # @Input1: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @Input2: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            self.current_time_slot_pooled_data_SBD_cumulated \
                = current_time_data + self.current_time_slot_pooled_data_SBD_cumulated

            # @ChangeShape: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # => [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            self.current_time_slot_pooled_data_SBD_cumulated \
                = self.current_time_slot_pooled_data_SBD_cumulated.permute(1, 0, 2)

            # Normalization Layer
            # @Input: [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            # @Output: [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            self.current_time_slot_pooled_data_SBD_cumulated \
                = self.after_time_attention_batch_normalization(self.current_time_slot_pooled_data_SBD_cumulated)

            print("@forward::ImprovedModelWithTimeSlot::Finish Normalization Layer")

            # @ChangeShape: [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
            # => [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            self.current_time_slot_pooled_data_SBD_cumulated \
                = self.current_time_slot_pooled_data_SBD_cumulated.permute(1, 0, 2)

            # Time Difference Sequence LSTM Layer
            # @InputRequired: [sequence_length, batch_size, dimension_reflected_to]
            # @Output: [sequence_length, batch_size, dimension_reflected_to]
            # @InputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            # @OutputActual: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
            self.current_time_slot_pooled_data_SBD_cumulated, \
            (self.lstm_time_recall_but_target_from_current_layer_hidden_state,
             self.lstm_time_recall_but_target_from_current_layer_cell_state) = \
                self.lstm_time_recall_but_target_from_current_layer(
                    self.current_time_slot_pooled_data_SBD_cumulated,
                    (self.lstm_time_recall_but_target_from_current_layer_hidden_state,
                     self.lstm_time_recall_but_target_from_current_layer_cell_state)
                )

            print("@forward::ImprovedModelWithTimeSlot::Finish Time Difference Sequence LSTM Layer")

            # 如果没有关系，就不需要再往后看了
            if current_time_data_max_value < 0.01:
                break

            # 超过最大回溯时间
            if current_time_slot_index >= self.max_time_slot:
                break



        # LSTM 输出的值域在 -1 到 1 之间，0代表没有关系，1代表有关系，-1代表负相关，相当于从记忆当中提取，然后修改当前的数据
        # 加到当前的数据上
        # @Input1: [sequence_length, batch_size, dimension_reflected_to]
        # @Input2: [sequence_length, batch_size, dimension_reflected_to]
        # @Output: [sequence_length, batch_size, dimension_reflected_to]
        output: torch.Tensor = current_time_data + self.tanh_reflected_to(
            self.current_time_slot_pooled_data_SBD_cumulated)

        print("@forward::ImprovedModelWithTimeSlot::Finish Tanh Layer")

        # @ChangeShape: [graph_height * graph_width, batch_size, feature_number * extracted_graph_feature]
        # => [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
        output = output.permute(1, 0, 2)

        # @ChangeShape: [batch_size, graph_height * graph_width, feature_number * extracted_graph_feature]
        # => [batch_size, feature_number * extracted_graph_feature * graph_height * graph_width]
        output = output.view(self.batch_size,
                             self.feature_number * self.cnn_conv_extracted_number
                             * self.new_graph_height * self.new_graph_width)

        # Final Fully Connected Layer
        # @InputRequired: [batch_size, input_size]
        # @Output: [batch_size, output_size]
        # @InputActual: [batch_size, feature_number * extracted_graph_feature * graph_height * graph_width]
        # @OutputActual: [batch_size, graph_height * graph_width]
        output = self.final_full_connection_layer(output)

        # @ChangeShape: [batch_size, graph_height * graph_width]
        # => [batch_size, graph_height, graph_width]
        output = output.view(self.batch_size,
                             self.graph_height,
                             self.graph_width)

        # @Output: [batch_size, graph_height, graph_width]

        print("@Forward::ImprovedModelWithTimeSlot::Finish Final Fully Connected Layer")

        # 重置clc，以及LSTM
        self.current_time_slot_index = 0
        self.reinit_lstm_time_attention_layer_hidden_state()
        self.reinit_lstm_time_recall_but_target_from_current_layer_hidden_state()
        self.lstmMultiheadAttention.reinit_hidden_cell_state()

        print("@Forward::ImprovedModelWithTimeSlot::Finish Resetting the Hidden State")
        print("==================================== "
              "@Forward::ImprovedModelWithTimeSlot::FINISH "
              "====================================")

        return output


if __name__ == '__main__':
    # 初始化测试
    model = ImprovedModelWithTimeSlot(batch_size=1, max_time_slot=10, feature_number=15, graph_height=540,
                                      graph_width=921, cnn_conv_stride=1, attention_head_number=2,
                                      lstm_time_attention_layer_number=2, final_linear_layer_hidden_size=25 * 25,
                                      data_type=torch.float32, device=torch.device('cpu'))
