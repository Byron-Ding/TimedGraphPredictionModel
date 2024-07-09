import torch


class FeatureBatchNormalization(torch.nn.Module):

    def __init__(self,
                 feature_layer_number: int,
                 graph_height: int,
                 graph_width: int,
                 device: torch.device,
                 ) -> None:
        super(FeatureBatchNormalization, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.device: torch.device = device
        self.feature_layer_number: int = feature_layer_number
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width

        # ========================== Normalization of different Layers ==========================
        # Normalization for reflecting rate for all features
        # Use 2D for both height and width
        self.feature_normalization = torch.nn.BatchNorm2d(
            num_features=feature_layer_number,
            eps=1e-05,  # Default
            momentum=0.1,  # Default
            affine=True,  # Default
            track_running_stats=True  # Default
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor: [batch_size, feature_number, graph_height, graph_width]
        # Normalization last two dimensions
        normalized_tensor: torch.Tensor = self.feature_normalization(input_tensor)

        # reflecting to
        normalized_tensor_height: torch.Tensor = torch.nn.functional.normalize(input=normalized_tensor,
                                                                                 p=2,
                                                                                 dim=-2)
        normalized_tensor_all: torch.Tensor = torch.nn.functional.normalize(input=normalized_tensor,
                                                                            p=2,
                                                                            dim=-1)
        return normalized_tensor_all


import torch


class FullFeatureNormalization(torch.nn.Module):

    def __init__(self,
                 total_layer_number: int,
                 graph_height: int,
                 graph_width: int,
                 device: torch.device,
                 ) -> None:
        super(FullFeatureNormalization, self).__init__()
        # ========================== Parameters BackUp ==========================
        self.device: torch.device = device
        self.total_layer_number: int = total_layer_number
        self.graph_height: int = graph_height
        self.graph_width: int = graph_width

        # ========================== Normalization of different Layers ==========================
        # Normalization for reflecting rate for all features
        # Use 2D for both height and width
        self.feature_normalization = torch.nn.BatchNorm2d(
            num_features=total_layer_number,
            eps=1e-05,  # Default
            momentum=0.1,  # Default
            affine=True,  # Default
            track_running_stats=True  # Default
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor: [batch_size, feature_number, graph_height, graph_width]
        # Normalization last two dimensions
        return self.feature_normalization(input_tensor)
