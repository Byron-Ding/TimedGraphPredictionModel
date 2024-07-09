import torch
from typing import Final


class PoolingLayer(torch.nn.Module):
    def __init__(self,
                 batch_size: int,
                 feature_number: int,
                 height_number: int,
                 width_number: int,
                 device: torch.device,
                 dtype: torch.dtype
                 ) -> None:
        super(PoolingLayer, self).__init__()
        self.batch_size: int = batch_size
        self.feature_number: int = feature_number
        self.height_number: int = height_number
        self.width_number: int = width_number

        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        self.pooling_divide: Final[int] = 12

        # ================================== Pooling Layer ==================================
        # 不pooling 太大了，数据太多了
        # [batch_size, feature_number, height, width]
        self.pooling_layer_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                                  stride=(2, 2),
                                                  padding=0)

        self.pooling_layer_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                                  stride=(2, 2),
                                                  padding=0)

        self.pooling_layer_3 = torch.nn.MaxPool2d(kernel_size=(3, 3),
                                                  stride=(3, 3),
                                                  padding=0)

        # self.pooling_layer_4 = torch.nn.MaxPool2d(kernel_size=(2, 2),
        #                                            stride=(2, 2),
        #                                            padding=0)

        # self.pooling_layer_5 = torch.nn.MaxPool2d(kernel_size=(2, 2),
        #                                           stride=(2, 2),
        #                                           padding=0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # [batch_size, feature_number, height, width]
        # [batch_size, feature_number, height/32, width/32]

        output1 = self.pooling_layer_1(input_tensor)
        output2 = self.pooling_layer_2(output1)
        output3 = self.pooling_layer_3(output2)
        # output4 = self.pooling_layer_4(output3)
        # output5 = self.pooling_layer_5(output4)
        return output3
