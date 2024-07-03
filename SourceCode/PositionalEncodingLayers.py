from TimedGraphPredictionModel.TimedGraphPredictionModel.SourceCode.SelfDefinedError import *


class PositionalEncoding_direct_mapped(torch.nn.Module):
    def __init__(self,
                 graph_height: int,
                 graph_width: int,
                 ):
        super(PositionalEncoding_direct_mapped, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width

    @staticmethod
    def forward(x: torch.Tensor,
                position_label: torch.Tensor,
                ) -> torch.Tensor:
        # x [batch_size, Dimension_reflected_to, H, W]
        # satellite_ground_cosine_graph [batch_size, H, W]

        # @ChangeShape: [batch_size, H, W]
        # => [batch_size, 1, H, W]
        position_label = position_label.unsqueeze(1)

        # ADD H W label
        x = x + position_label

        return x


class PositionalEncoding_FeatureLabel(torch.nn.Module):
    def __init__(self,
                 feature_number: int,
                 graph_height: int,
                 graph_width: int,
                 ):
        super(PositionalEncoding_FeatureLabel, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.feature_number: int = feature_number
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width

    @staticmethod
    def forward(x: torch.Tensor,
                feature_number: torch.Tensor,
                ) -> torch.Tensor:
        # x [batch_size, feature_number, graph_height_range, graph_width_range]
        # feature_number [batch_size, feature_number]
        expected_feature_number: int = feature_number.size(1)
        input_feature_number: int = x.size(1)

        # 扩展维度
        # [batch_size, feature_number]
        # => [batch_size, feature_number, graph_height_range(1), graph_width_range(1)]
        if input_feature_number != expected_feature_number:
            raise ValueError(f"Expect {feature_number.shape}[{expected_feature_number}] feature number "
                             f"but got {x.shape}[{input_feature_number}] feature")

        feature_number = feature_number.unsqueeze(2).unsqueeze(3)

        # 相加 PLUS
        x = x + feature_number

        return x


class PositionalEncoding_Height_Width(torch.nn.Module):

    def __init__(self,
                 dimension_reflected_to: int,
                 device: torch.device,
                 max_len: int = 100000,
                 base: int = 100000000,
                 dtype: torch.dtype = torch.double,
                 ):
        """
        Forward: [batch_size, seq_len, dimension_reflected_to]
        :param dimension_reflected_to:
        :param device:
        :param max_len:
        :param base:
        """
        super(PositionalEncoding_Height_Width, self).__init__()

        # ========================== Parameters BackUp ==========================
        self.device = device
        self.dtype = dtype

        # 输入 [batch_size, feature_numbers, graph_height_range * graph_width_range, input_dimension_reflected_to]
        # graph_height_range * graph_width_range, = 540*921 = 497340
        # 按行拼接的

        # 位置编码
        # 作用是为输入的词向量添加位置信息

        # 预先定义位置编码矩阵
        # 字典大小为 max_len
        # 词位置向量维度为 input_dimension_reflected_to
        # [batch_size, max_len, input_dimension_reflected_to]
        '''
        最后
        [sin(1), sin(2), ..., sin(input_dimension_reflected_to)]
        [cos(1), cos(2), ..., cos(input_dimension_reflected_to)]
        [sin(2), sin(3), ..., sin(input_dimension_reflected_to + 1)]
        '''
        # [max_len, input_dimension_reflected_to]
        position_encoding_matrix: torch.Tensor = torch.zeros(max_len,
                                                             dimension_reflected_to,
                                                             dtype=dtype) \
            .to(device=device)

        # i 是 变化的 input_dimension_reflected_to 的序数 : torch.Tensor
        # input_dimension_reflected_to 是 总的维数的大小 (int)
        # PE(pos, 2i) = sin(pos / 10000^(2i / input_dimension_reflected_to))
        # PE(pos, 2i + 1) = cos(pos / 10000^(2i / input_dimension_reflected_to))
        # 生成 pos 序列
        # [0, ..., max_len - 1]
        # [input_index]
        input_position_index: torch.Tensor = torch.arange(0, max_len, dtype=dtype).to(device=device)

        # 生成序列 i (因为sin和cos是成对的)
        # [0, ..., input_dimension_reflected_to - 1]
        dimension_reflected_to_index: torch.Tensor \
            = torch.arange(0, dimension_reflected_to, dtype=dtype) \
            .to(device=device)

        # 计算 10000^(2 Current_dimension / NUM_dimension_reflected_to)
        # 长度为 [input_dimension_reflected_to]
        main_divider: torch.Tensor = torch.pow(
            base,
            2 * dimension_reflected_to_index / dimension_reflected_to
        ).to(device=device)

        # 1 / main_divider
        main_divider = 1 / main_divider
        # => [1, input_dimension_reflected_to] (为后面的除法/矩阵乘法做准备)
        main_divider = main_divider.unsqueeze(0)

        # input_position_index.size() = [max_len]
        # input_position_index.unsqueeze(0).size() = [max_len, 1]
        input_position_index = input_position_index.unsqueeze(1)

        # print(main_divider.size())
        # print(input_position_index.size())
        # print(max_len, input_dimension_reflected_to)

        # [max_len, 1]
        # * [1, input_dimension_reflected_to]
        # => [max_len, input_dimension_reflected_to]
        inside_trigonometric: torch.Tensor = torch.matmul(input_position_index, main_divider).to(device=self.device)

        # 生成 sin 和 cos 的位置编码
        # [max_len, input_dimension_reflected_to]
        # input_position_index.size() = [max_len, 1]
        # main_divider.size() = [1, input_dimension_reflected_to]
        # torch.matmul(input_position_index, main_divider).size() = [max_len, input_dimension_reflected_to]
        inside_trigonometric_general: torch.Tensor = inside_trigonometric[max_len // 2]

        # 偶数位置
        position_encoding_matrix[0::2, :] = torch.sin(inside_trigonometric_general).to(device=device)
        # 奇数位置
        position_encoding_matrix[1::2, :] = torch.cos(inside_trigonometric_general).to(device=device)
        # 补齐 奇数个数情况
        if position_encoding_matrix.size()[0] % 2 == 1:
            position_encoding_matrix[-1, :] = torch.sin(inside_trigonometric[max_len // 2 + 1]).to(device=device)
        # 这样对于输入序列不同的位置，就有不同的sin/cos序列

        position_encoding_matrix: torch.Tensor = position_encoding_matrix.to(device=device)
        # 位置编码矩阵 存储，不参与训练
        # [max_len, input_dimension_reflected_to]
        self.register_buffer("position_encoding_matrix", position_encoding_matrix)
        # => [max_len, input_dimension_reflected_to]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 把所有层
        # 输入 [batch_size * feature_number, graph_height_range, graph_width_range]
        # position_encoding_matrix [max_len, input_dimension_reflected_to]

        # @Input: [batch_size, seq_len, dimension_reflected_to]
        # self.position_encoding_matrix: torch.tensor = [max_len, dimension_reflected_to]
        fitted_position_encoding_matrix: torch.tensor = self.position_encoding_matrix.unsqueeze(0)

        # add them together
        x = x + fitted_position_encoding_matrix[:, :x.size(1), :]

        # @Output: [batch_size, seq_len, dimension_reflected_to]
        return x



class PositionalEncoding_AllFeatures(torch.nn.Module):
    def __init__(self,
                 feature_number: int,
                 graph_height: int,
                 graph_width: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        super(PositionalEncoding_AllFeatures, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.feature_number: int = feature_number
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        # @Input: [batch_size, feature_number, graph_height, graph_width]

        # ========================== Positional Encoding ==========================
        # -------------------------- graph_height graph_width --------------------------
        # 位置编码 graph_height graph_width
        self.position_encoding_height_width = PositionalEncoding_Height_Width(
            dimension_reflected_to=graph_width,
            device=device,
            max_len=graph_height,
            base=graph_height,
            dtype=dtype
        )

        # -------------------------- Cosine Position Encoding --------------------------
        # 位置编码 Satellite Ground
        # Satellite Ground
        self.position_encoding_satellite_ground_cosine = PositionalEncoding_direct_mapped(
            graph_height=graph_height,
            graph_width=graph_width
        )

        # -------------------------- feature label Position Encoding --------------------------
        self.feature_label_position: PositionalEncoding_FeatureLabel = PositionalEncoding_FeatureLabel(
            feature_number=feature_number,
            graph_height=graph_height,
            graph_width=graph_width
        )

    def forward(self,
                x: torch.Tensor,
                satellite_ground_cosine_graph: torch.Tensor,
                feature_number: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:
        x_batch_size: int = x.size(0)

        # x [batch_size, feature_number, graph_height, graph_width]
        # satellite_ground_cosine_graph [batch_size, graph_height, graph_width]
        # feature_number [batch_size, feature_number]
        # @Output: [batch_size, feature_number, graph_height, graph_width]

        # ========================== 2D->1D Position Encoding ==========================
        # -------------------------- Preparation --------------------------
        # Position Encoding graph_height graph_width
        # @Input: [batch_size, seq_len, dimension_reflected_to]
        # self.position_encoding_matrix: torch.tensor = [max_len, dimension_reflected_to]
        # @ChangeShape: [batch_size, feature_number, graph_height, graph_width]
        # => [batch_size * feature_number, graph_height, graph_width]
        x = x.view(x_batch_size * self.feature_number, self.graph_height, self.graph_width)

        # -------------------------- graph_height graph_width Position Encoding --------------------------
        # 位置编码 graph_height graph_height
        # @InputRequired [batch_size, seq_len, dimension_reflected_to]
        x_height_width = self.position_encoding_height_width(x)
        # => [batch_size, seq_len, dimension_reflected_to]

        # @ChangeShape: [batch_size, seq_len, dimension_reflected_to]
        # => [batch_size, feature_number, graph_height, graph_width]
        x_height_width = x_height_width.view(x_batch_size,  # batch_size
                                             self.feature_number,
                                             self.graph_height,
                                             self.graph_width)
        # => [batch_size, feature_number, graph_height, graph_width]

        # ========================== Cosine Position Encoding ==========================
        # @InputRequired: [batch_size, feature_number, graph_height, graph_width]
        # @InputRequired: [batch_size, graph_height, graph_width]
        x_height_width_plus_direct_label: torch.Tensor = self.position_encoding_satellite_ground_cosine(
            x_height_width,
            satellite_ground_cosine_graph)
        # => [batch_size, feature_number, graph_height, graph_width]

        # ========================== Feature Number Position Encoding ==========================
        # @InputRequired: [batch_size, feature_number, graph_height, graph_height]
        # @InputRequired: [batch_size, feature_number]
        x_height_width_plus_direct_label: torch.Tensor \
            = self.feature_label_position(
                x_height_width_plus_direct_label,
                feature_number
            )
        # @Output: [batch_size, feature_number, graph_height, graph_width]

        return x_height_width_plus_direct_label


class PositionalEncoding_TimeStamp(torch.nn.Module):

    def __init__(self,
                 ):
        super(PositionalEncoding_TimeStamp, self).__init__()


        # ========================== Parameters BackUp ==========================


    @staticmethod
    def forward(x: torch.Tensor, time_stamp: float) -> torch.Tensor:
        x = x + time_stamp

        return x
