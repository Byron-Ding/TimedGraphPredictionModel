import torch
import typing


class CNNFeatureExtractionNetwork(torch.nn.Module):
    # I tensorflow/core/util/port.cc:113] oneDNN custom operations are on.
    # You may see slightly different numerical results
    # due to floating-point round-off errors from different computation orders.
    # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    # 这个是因为TF2.0默认开启了oneDNN，可以通过设置环境变量关闭
    # export TF_ENABLE_ONEDNN_OPTS=0
    # 可以不用管，因为浮点数处理会有轻微数值结果漂移，
    def __init__(self,
                 batch_size: int,
                 feature_number: int,
                 height: int,
                 width: int,
                 cnn_conv_stride: int,
                 device: torch.device,
                 dtype: torch.dtype
                 ) -> None:
        super(CNNFeatureExtractionNetwork, self).__init__()

        # ================================== Parameter Backup ==================================
        self.batch_size: int = batch_size
        self.feature_number: int = feature_number
        self.cnn_conv_stride: int = cnn_conv_stride

        self.width: int = width
        self.height: int = height
        self.original_area: int = width * height

        self.device: torch.device = device
        self.dtype = dtype

        print("@Init::CNNFeatureExtractionNetwork::Finish Init Parameter Backup")

        # ================================== Feature Extraction ==================================
        """
        3x3, 7x7, 15x15, 31x31, 63x63, 127x127
        """
        """
        GPT の 意见 
        感受野 15：使用  5 × 5
        5×5 卷积核，3 层。
        感受野 31：使用  5 × 5
        5×5 卷积核，7 层。
        感受野 63：使用  5 × 5
        5×5 卷积核，15 层。
        感受野 127：使用 7 × 7
        7×7 卷积核，21 层。
        """
        self.layer_5x5_simulate_15x15: typing.Final[int] = 3
        self.layer_5x5_simulate_31x31: typing.Final[int] = 7
        self.layer_5x5_simulate_63x63: typing.Final[int] = 15
        self.layer_7x7_simulate_127x127: typing.Final[int] = 21


        # [batch_size, input_dimension, H, W]
        # 2D Convolutional Neural Network 3x3
        # @ChangeShape: [batch_size, input_dimension + 1, H, W]
        # => [batch_size, input_dimension + 1, H, W]
        self.convolutional_neural_network_3x3: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.feature_number,
            out_channels=self.feature_number,
            kernel_size=3,
            stride=self.cnn_conv_stride,
            padding=1,
            dtype=self.dtype, )

        self.convolutional_neural_network_5x5: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.feature_number,
            out_channels=self.feature_number,
            kernel_size=5,
            stride=self.cnn_conv_stride,
            padding=2,
            dtype=self.dtype, )

        # 2D Convolutional Neural Network 4x4
        self.convolutional_neural_network_7x7: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.feature_number,
            out_channels=self.feature_number,
            kernel_size=7,
            stride=self.cnn_conv_stride,
            padding=3,
            dtype=self.dtype, )

        self.convolutional_neural_network_15x15_simulation: list[torch.nn.Conv2d] \
            = [torch.nn.Conv2d(
                                in_channels=self.feature_number,
                                out_channels=self.feature_number,
                                kernel_size=5,
                                stride=self.cnn_conv_stride,
                                padding=2,
                                dtype=self.dtype, ) for _ in range(self.layer_5x5_simulate_15x15)]

        self.convolutional_neural_network_15x15: torch.nn.Sequential = torch.nn.Sequential(
            *self.convolutional_neural_network_15x15_simulation)

        self.convolutional_neural_network_31x31_simulation: list[torch.nn.Conv2d] \
            = [torch.nn.Conv2d(
                                in_channels=self.feature_number,
                                out_channels=self.feature_number,
                                kernel_size=5,
                                stride=self.cnn_conv_stride,
                                padding=2,
                                dtype=self.dtype, ) for _ in range(self.layer_5x5_simulate_31x31)]

        self.convolutional_neural_network_31x31: torch.nn.Sequential = torch.nn.Sequential(
            *self.convolutional_neural_network_31x31_simulation)

        self.convolutional_neural_network_63x63_simulation: list[torch.nn.Conv2d] \
            = [torch.nn.Conv2d(
                                in_channels=self.feature_number,
                                out_channels=self.feature_number,
                                kernel_size=5,
                                stride=self.cnn_conv_stride,
                                padding=2,
                                dtype=self.dtype, ) for _ in range(self.layer_5x5_simulate_63x63)]

        self.convolutional_neural_network_63x63: torch.nn.Sequential = torch.nn.Sequential(
            *self.convolutional_neural_network_63x63_simulation)

        self.convolutional_neural_network_127x127_simulation: list[torch.nn.Conv2d] \
            = [torch.nn.Conv2d(
                                in_channels=self.feature_number,
                                out_channels=self.feature_number,
                                kernel_size=7,
                                stride=self.cnn_conv_stride,
                                padding=3,
                                dtype=self.dtype, ) for _ in range(self.layer_7x7_simulate_127x127)]

        self.convolutional_neural_network_127x127: torch.nn.Sequential = torch.nn.Sequential(
            *self.convolutional_neural_network_127x127_simulation)

        print("@Init::CNNFeatureExtractionNetwork::Finish Init Convolutional Neural Network")

        # ================================== Batch Normalization ==================================
        # @Input: [B, C, H, W]
        self.batch_norm_3x3: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_5x5: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_7x7: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_15x15: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_31x31: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_63x63: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.batch_norm_127x127: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            num_features=self.feature_number,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)

        print("@Init::CNNFeatureExtractionNetwork::Finish Init Batch Normalization")

        # ================================== Activation Function ==================================
        self.activation_function_3x3: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_5x5: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_7x7: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_15x15: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_31x31: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_63x63: torch.nn.SELU = torch.nn.SELU()
        self.activation_function_127x127: torch.nn.SELU = torch.nn.SELU()

        print("@Init::CNNFeatureExtractionNetwork::Finish Init Activation Function")

        print("@Init::CNNFeatureExtractionNetwork::Finish Init CNN Part")

    def forward(self,
                input_tensor: torch.Tensor,
                ) -> torch.Tensor:
        # input_tensor: [batch_size, feature_number, H, W]
        # output_tensor: [batch_size, feature_number, H, W, extracted_graph_feature]
        print("@Forward::CNNFeatureExtractionNetwork::Start Forward")

        # ================================== Feature Extraction ==================================
        extracted_tensor_3x3: torch.Tensor = self.convolutional_neural_network_3x3(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 3x3 Convolutional Neural Network")
        # print(extracted_tensor_3x3.shape)
        extracted_tensor_5x5: torch.Tensor = self.convolutional_neural_network_5x5(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 5x5 Convolutional Neural Network")
        # print(extracted_tensor_5x5.shape)
        extracted_tensor_7x7: torch.Tensor = self.convolutional_neural_network_7x7(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 7x7 Convolutional Neural Network")
        # print(extracted_tensor_7x7.shape)
        extracted_tensor_15x15: torch.Tensor = self.convolutional_neural_network_15x15(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 15x15 Convolutional Neural Network")
        # print(extracted_tensor_15x15.shape)
        extracted_tensor_31x31: torch.Tensor = self.convolutional_neural_network_31x31(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 31x31 Convolutional Neural Network")
        # print(extracted_tensor_31x31.shape)
        extracted_tensor_63x63: torch.Tensor = self.convolutional_neural_network_63x63(input_tensor)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 63x63 Convolutional Neural Network")
        # print(extracted_tensor_63x63.shape)
        extracted_tensor_127x127: torch.Tensor = self.convolutional_neural_network_127x127(input_tensor)
        # print(extracted_tensor_127x127.shape)
        print("@Forward::CNNFeatureExtractionNetwork::Finish 127x127 Convolutional Neural Network")

        print("Finish Feature Extraction")

        # ================================== Batch Normalization ==================================
        # @Input: [B, C, H, W]
        # @Output: [B, C, H, W]
        normalized_tensor_3x3: torch.Tensor = self.batch_norm_3x3(extracted_tensor_3x3)
        normalized_tensor_5x5: torch.Tensor = self.batch_norm_5x5(extracted_tensor_5x5)
        normalized_tensor_7x7: torch.Tensor = self.batch_norm_7x7(extracted_tensor_7x7)
        normalized_tensor_15x15: torch.Tensor = self.batch_norm_15x15(extracted_tensor_15x15)
        normalized_tensor_31x31: torch.Tensor = self.batch_norm_31x31(extracted_tensor_31x31)
        normalized_tensor_63x63: torch.Tensor = self.batch_norm_63x63(extracted_tensor_63x63)
        normalized_tensor_127x127: torch.Tensor = self.batch_norm_127x127(extracted_tensor_127x127)

        print("Finish Batch Normalization")

        # ================================== Activation Function ==================================
        activated_tensor_3x3: torch.Tensor = self.activation_function_3x3(normalized_tensor_3x3)
        activated_tensor_5x5: torch.Tensor = self.activation_function_5x5(normalized_tensor_5x5)
        activated_tensor_7x7: torch.Tensor = self.activation_function_7x7(normalized_tensor_7x7)
        activated_tensor_15x15: torch.Tensor = self.activation_function_15x15(normalized_tensor_15x15)
        activated_tensor_31x31: torch.Tensor = self.activation_function_31x31(normalized_tensor_31x31)
        activated_tensor_63x63: torch.Tensor = self.activation_function_63x63(normalized_tensor_63x63)
        activated_tensor_127x127: torch.Tensor = self.activation_function_127x127(normalized_tensor_127x127)

        print("Finish Activation Function")

        # 最后拼接 所有的特征为新的维度
        # @ChangeShape: [batch_size, feature_number, H, W]
        # => [batch_size, feature_number, H, W, extracted_graph_feature]
        output_tensor: torch.Tensor = torch.stack((
            activated_tensor_3x3,
            activated_tensor_5x5,
            activated_tensor_7x7,
            activated_tensor_15x15,
            activated_tensor_31x31,
            activated_tensor_63x63,
            activated_tensor_127x127
        ), dim=4)

        return output_tensor


if __name__ == '__main__':
    # Initialize the model
    model = CNNFeatureExtractionNetwork(batch_size=1, feature_number=15, height=540, width=921, cnn_conv_stride=1,
                                        device=torch.device('cuda:0'), dtype=torch.float32)
