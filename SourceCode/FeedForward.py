import gc

import torch


class FeedForward(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 head_number: int,
                 device: torch.device,
                 ):
        # 非线性 # 我们需要负数，所以不用relu, 使用 swish
        super().__init__()
        self.linear_forward_layer1 = torch.nn.Linear(head_number,
                                                     hidden_size,
                                                     bias=False).to(device=device)
        self.non_linear_layer = torch.nn.SELU()
        self.linear_forward_layer2 = torch.nn.Linear(hidden_size,
                                                     hidden_size,
                                                     bias=False).to(device=device)
        self.non_linear_layer2 = torch.nn.SELU()
        self.linear_forward_layer3 = torch.nn.Linear(hidden_size,
                                                     head_number,
                                                     bias=False).to(device=device)

        self.feed_forward_layer = torch.nn.Sequential(
            self.linear_forward_layer1,
            self.non_linear_layer,
            self.linear_forward_layer2,
            self.non_linear_layer2,
            self.linear_forward_layer3
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.feed_forward_layer(input_tensor)

        gc.collect()
        torch.cuda.empty_cache()
        return output