from SelfDefinedError import *


class PositionalEncoding_OtherPixelLabel(torch.nn.Module):
    def __init__(self,
                 height_number: int,
                 width_number: int,
                 ):
        super(PositionalEncoding_OtherPixelLabel, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.height_number: int = height_number
        self.width_number: int = width_number

    @staticmethod
    def forward(x: torch.Tensor,
                other_pixel_label_graph: torch.Tensor,
                ) -> torch.Tensor:
        # x [batch_size, feature_layer, height_range, width_range]
        # other_pixel_label_graph [batch_size, height_range, width_range]

        # @ChangeShape: [batch_size, height_range, width_range]
        # => [batch_size, 1, height_range, width_range]
        other_pixel_label_graph = other_pixel_label_graph.unsqueeze(1)

        # ADD  Cosine
        x = x + other_pixel_label_graph

        return x


class PositionalEncoding_FeatureLabel(torch.nn.Module):
    def __init__(self,
                 feature_number: int,
                 height_number: int,
                 width_number: int,
                 ):
        super(PositionalEncoding_FeatureLabel, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.feature_number: int = feature_number
        self.height_number: int = height_number
        self.width_number: int = width_number

    @staticmethod
    def forward(x: torch.Tensor,
                feature_feature: torch.Tensor,
                ) -> torch.Tensor:
        # x [batch_size, feature_layer, height_range, width_range]
        # feature_feature [batch_size, feature_layer]
        expected_feature_number: int = feature_feature.size(1)
        input_feature_number: int = x.size(1)

        # 扩展维度
        # [batch_size, feature_layer]
        # => [batch_size, feature_layer, height_range(1), width_range(1)]
        if input_feature_number != expected_feature_number:
            raise ValueError(f"Expect {feature_feature.shape}[{expected_feature_number}] feature number "
                             f"but got {x.shape}[{input_feature_number}] feature")

        feature_feature = feature_feature.unsqueeze(2).unsqueeze(3)

        # 相加 PLUS
        x = x + feature_feature

        return x


class PositionalEncoding_width_height(torch.nn.Module):

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
        super(PositionalEncoding_width_height, self).__init__()

        # ========================== Parameters BackUp ==========================
        self.device = device
        self.dtype = dtype

        # 输入 [batch_size, feature_layers, height_range * width_range, input_dimension_reflected_to]
        # height_range * width_range, = 540*921 = 497340
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
        # 输入 [batch_size * feature_layer, width_range, height_range]
        # position_encoding_matrix [max_len, input_dimension_reflected_to]

        # @Input: [batch_size, seq_len, dimension_reflected_to]
        # self.position_encoding_matrix: torch.tensor = [max_len, dimension_reflected_to]
        fitted_position_encoding_matrix: torch.tensor = self.position_encoding_matrix.unsqueeze(0)

        # add them together
        x = x + fitted_position_encoding_matrix[:, :x.size(1), :]

        # @Output: [batch_size, seq_len, dimension_reflected_to]
        return x


class PositionalEncoding_feature(torch.nn.Module):

    def __init__(self,
                 device: torch.device,
                 height_number: int,
                 width_number: int,
                 feature_number: int,
                 dtype: torch.dtype,
                 max_len: int = 100000,
                 base: int = 10000,
                 ):
        super(PositionalEncoding_feature, self).__init__()

        # ========================== Parameters BackUp ==========================
        self.device: torch.device = device
        self.height_number: int = height_number
        self.width_number: int = width_number
        self.feature_layer_number: int = feature_number
        self.max_len: int = max_len
        self.base: int = base
        self.dtype: torch.dtype = dtype

        # ========================== Positional Encoding ==========================
        # -------------------------- 2D->1D Position Encoding --------------------------
        # 位置编码 height width/使用Sin和Cos
        self.position_encoding_height = PositionalEncoding_width_height(
            dimension_reflected_to=width_number,
            device=device,
            max_len=max_len,
            base=base,
            dtype=dtype
        )

        # -------------------------- Cosine Position Encoding --------------------------
        # 位置编码 
        # 
        self.position_encoding_other_pixel_label = PositionalEncoding_OtherPixelLabel(
            height_number=height_number,
            width_number=width_number
        )

        # -------------------------- feature feature Position Encoding --------------------------
        # 位置编码 feature feature
        # feature feature
        self.position_encoding_feature_feature = PositionalEncoding_FeatureLabel(
            feature_number=feature_number,
            height_number=height_number,
            width_number=width_number
        )

    def forward(self,
                x: torch.Tensor,
                feature_feature: torch.Tensor,
                other_pixel_label_graph: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:
        # x [batch_size, feature_layer, height_range, width_range]

        # ========================== 2D->1D Position Encoding ==========================
        # -------------------------- Preparation --------------------------

        # -------------------------- height width Position Encoding --------------------------
        # 位置编码 height width
        # [batch_size, feature_layer, height_range, width_range]
        x_height_width = self.position_encoding_height(x)
        # => [batch_size, feature_layer, height_range, width_range]

        # -------------------------- Cosine Position Encoding --------------------------
        x_height_width = self.position_encoding_other_pixel_label(x_height_width,
                                                                              other_pixel_label_graph)
        # => [batch_size, feature_layer, height_range, width_range]

        # -------------------------- feature feature Position Encoding --------------------------
        x_height_width = self.position_encoding_feature_feature(x_height_width, feature_feature)
        # => [batch_size, feature_layer, height_range, width_range]

        return x_height_width


class PositionalEncoding_GeometricSurface(torch.nn.Module):
    def __init__(self,
                 height_number: int,
                 width_number: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        super(PositionalEncoding_GeometricSurface, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.height_number: int = height_number
        self.width_number: int = width_number

        # ========================== Positional Encoding ==========================
        # -------------------------- height width --------------------------
        # 位置编码 height width
        self.position_encoding_height = PositionalEncoding_width_height(
            dimension_reflected_to=width_number,
            device=device,
            max_len=height_number,
            base=height_number,
            dtype=dtype
        )

        # -------------------------- Cosine Position Encoding --------------------------
        # 位置编码 
        # 
        self.position_encoding_other_pixel_label = PositionalEncoding_OtherPixelLabel(
            height_number=height_number,
            width_number=width_number
        )

    def forward(self,
                x: torch.Tensor,
                other_pixel_label_graph: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:
        # x [batch_size, feature_number, height_range, width_range]
        # ========================== 2D->1D Position Encoding ==========================
        # -------------------------- Preparation --------------------------

        # -------------------------- height width Position Encoding --------------------------
        # 位置编码 height width
        # height
        # [batch_size, height_range, width_range]
        x_height_width = self.position_encoding_height(x)
        # => [batch_size, height_range, width_range]

        # ========================== Cosine Position Encoding ==========================
        x_height_width = self.position_encoding_other_pixel_label(x_height_width,
                                                                              other_pixel_label_graph)
        # => [batch_size, height_range, width_range]

        return x_height_width


class PositionalEncoding_VisibleInfraredGeometric(torch.nn.Module):
    def __init__(self,
                 total_layer_number: int,
                 height_number: int,
                 width_number: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        super(PositionalEncoding_VisibleInfraredGeometric, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.total_layer_number: int = total_layer_number
        self.height_number: int = height_number
        self.width_number: int = width_number
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        # ========================== Positional Encoding ==========================
        # -------------------------- height width --------------------------
        # 位置编码 height width
        self.position_encoding_height = PositionalEncoding_width_height(
            dimension_reflected_to=width_number,
            device=device,
            max_len=height_number,
            base=height_number,
            dtype=dtype
        )

        # -------------------------- Cosine Position Encoding --------------------------
        # 位置编码 
        # 
        self.position_encoding_other_pixel_label = PositionalEncoding_OtherPixelLabel(
            height_number=height_number,
            width_number=width_number
        )

        # -------------------------- feature feature Position Encoding --------------------------
        # 位置编码 feature feature and other labels
        self.position_encoding_feature_feature = PositionalEncoding_FeatureLabel(
            feature_number=total_layer_number,
            height_number=height_number,
            width_number=width_number
        )

    def forward(self,
                x: torch.Tensor,
                wave_length_and_other_labels: torch.Tensor,
                other_pixel_label_graph: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:
        # x [batch_size, feature_layer, height_range, width_range]
        # other_pixel_label_graph [batch_size, height_range, width_range]
        # wave_length_and_other_labels [batch_size, feature_layer]

        # ========================== 2D->1D Position Encoding ==========================
        # -------------------------- Preparation --------------------------

        # -------------------------- height width Position Encoding --------------------------
        # 位置编码 height width
        # [batch_size, feature_layer, height_range, width_range]
        x_height_width = self.position_encoding_height(x)
        # => [batch_size, feature_layer, height_range, width_range]

        # ========================== Cosine Position Encoding ==========================
        x_height_width = self.position_encoding_other_pixel_label(x_height_width,
                                                                              other_pixel_label_graph)
        # => [batch_size, feature_layer, height_range, width_range]
        # ========================== feature feature Position Encoding ==========================
        # [batch_size, feature_layer, height_range, width_range]
        # [batch_size, feature_layer]
        # => [batch_size, feature_layer, height_range, width_range]
        x_height_width \
            = self.position_encoding_feature_feature(x_height_width, wave_length_and_other_labels)

        return x_height_width


class PositionalEncoding_AllFeatures(torch.nn.Module):
    def __init__(self,
                 feature_number: int,
                 height_number: int,
                 width_number: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        super(PositionalEncoding_AllFeatures, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.feature_number: int = feature_number
        self.height_number: int = height_number
        self.width_number: int = width_number
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        # @Input: [batch_size, feature_number, graph_height, graph_width]

        # ========================== Positional Encoding ==========================
        # -------------------------- height width --------------------------
        # 位置编码 height width
        self.position_encoding_height_width = PositionalEncoding_width_height(
            dimension_reflected_to=width_number,
            device=device,
            max_len=height_number,
            base=height_number,
            dtype=dtype
        )

        # -------------------------- Cosine Position Encoding --------------------------
        # 位置编码 
        # 
        self.position_encoding_other_pixel_label = PositionalEncoding_OtherPixelLabel(
            height_number=height_number,
            width_number=width_number
        )

        # -------------------------- feature label Position Encoding --------------------------
        self.feature_label_position: PositionalEncoding_FeatureLabel = PositionalEncoding_FeatureLabel(
            feature_number=feature_number,
            height_number=height_number,
            width_number=width_number
        )

    def forward(self,
                x: torch.Tensor,
                other_pixel_label_graph: torch.Tensor,
                feature_feature: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:
        x_batch_size: int = x.size(0)

        # x [batch_size, feature_layer, height_range, width_range]
        # other_pixel_label_graph [batch_size, height_range, width_range]
        # feature_feature [batch_size, feature_layer]
        # @Output: [batch_size, feature_layer, height_range, width_range]

        # ========================== 2D->1D Position Encoding ==========================
        # -------------------------- Preparation --------------------------
        # Position Encoding height width
        # @Input: [batch_size, seq_len, dimension_reflected_to]
        # self.position_encoding_matrix: torch.tensor = [max_len, dimension_reflected_to]
        # @ChangeShape: [batch_size, feature_number, graph_height, graph_width]
        # => [batch_size * feature_number, graph_height, graph_width]
        x = x.view(x_batch_size * self.feature_number, self.height_number, self.width_number)

        # -------------------------- height width Position Encoding --------------------------
        # 位置编码 height width
        # @InputRequired [batch_size, seq_len, dimension_reflected_to]
        x_height_width = self.position_encoding_height_width(x)
        # => [batch_size, seq_len, dimension_reflected_to]

        # @ChangeShape: [batch_size, seq_len, dimension_reflected_to]
        # => [batch_size, feature_number, graph_height, graph_width]
        x_height_width = x_height_width.view(x_batch_size,  # batch_size
                                   self.feature_number,
                                   self.height_number,
                                   self.width_number)
        # => [batch_size, feature_layer, height_range, width_range]

        # ========================== Cosine Position Encoding ==========================
        # @InputRequired: [batch_size, feature_layer, height_range, width_range]
        # @InputRequired: [batch_size, height_range, width_range]
        x_height_width_cos: torch.Tensor = self.position_encoding_other_pixel_label(
            x_height_width,
            other_pixel_label_graph)
        # => [batch_size, feature_layer, height_range, width_range]

        # ========================== feature feature Position Encoding ==========================
        # @InputRequired: [batch_size, feature_layer, height_range, width_range]
        # @InputRequired: [batch_size, feature_layer]
        x_height_width_cos_label: torch.Tensor \
            = self.feature_label_position(
                x_height_width_cos,
                feature_feature
            )
        # @Output: [batch_size, feature_layer, height_range, width_range]

        return x_height_width_cos_label


class PositionalEncoding_TimeStamp(torch.nn.Module):

    def __init__(self,
                 ):
        super(PositionalEncoding_TimeStamp, self).__init__()


        # ========================== Parameters BackUp ==========================


    @staticmethod
    def forward(x: torch.Tensor, time_stamp: float) -> torch.Tensor:
        x = x + time_stamp

        return x
