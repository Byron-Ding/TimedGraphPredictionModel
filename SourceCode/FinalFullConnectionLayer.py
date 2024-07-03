import torch


class FinalFullConnectionLayer(torch.nn.Module):
    """
    The final full connection layer
    """

    def __init__(self,
                 batch_size: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 device: torch.device,
                 dtype: torch.dtype
                 ) -> None:
        super(FinalFullConnectionLayer, self).__init__()
        self.batch_size: int = batch_size
        self.feature_number: int = input_size

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.device: torch.device = device
        self.dtype = dtype

        # ================================== Full Connection Layer ==================================
        # [batch_size, feature_number, head_number]
        self.linear_layer_1 = torch.nn.Linear(self.feature_number,
                                              self.hidden_size)
        self.non_linear_layer_1_2 = torch.nn.SELU()
        self.linear_layer_2 = torch.nn.Linear(self.hidden_size,
                                              self.hidden_size)
        self.non_linear_layer_2_3 = torch.nn.SELU()
        self.linear_layer_3 = torch.nn.Linear(self.hidden_size,
                                              self.output_size)

        # xavier初始化
        torch.nn.init.xavier_normal_(self.linear_layer_1.weight)
        torch.nn.init.xavier_normal_(self.linear_layer_2.weight)
        torch.nn.init.xavier_normal_(self.linear_layer_3.weight)



    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 不用Sequential,因为这玩意会一次性把所有的层都放在一起，然后显存就爆了
        # [batch_size, feature_number]
        # [batch_size, output_size]
        output = self.linear_layer_1(input_tensor)
        output = self.non_linear_layer_1_2(output)
        output = self.linear_layer_2(output)
        output = self.non_linear_layer_2_3(output)
        output = self.linear_layer_3(output)
        return output

