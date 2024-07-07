import torch
import numpy


def create_mask(input_q: torch.Tensor,
                input_k: torch.Tensor) -> torch.Tensor:
    # 生成mask
    # 对角线 mask， 左下角为True，右上角为False
    if len(input_q.size()) == 4 or len(input_k.size()) == 4:
        # input_q [batch_size, seq_len, input_dimension_reflected_to]
        # input_k [batch_size, seq_len, input_dimension_reflected_to]
        column_number: int = input_q.size(2)
        row_number: int = input_k.size(2)

    else:
        # input_q [batch_size, seq_len, input_dimension_reflected_to]
        # input_k [batch_size, seq_len, input_dimension_reflected_to]
        column_number: int = input_q.size(1)
        row_number: int = input_k.size(1)

    # 创建一个全为 True 的矩阵
    mask = torch.ones(column_number, row_number, dtype=torch.bool)
    # 左上到右下，对角线，因为现在是Q为行，K为列，所以是上三角遮蔽
    mask = torch.triu(mask, diagonal=1)

    return mask


def test_nan(input_tensor: torch.Tensor) -> bool:
    # 判断是否有nan=
    a = torch.isnan(input_tensor)

    if a.any():
        raise Exception("Model output is All NaN values!")

    return False


def mask_nan_by_number(x: numpy.ma.core.MaskedArray,
                       mask_number: float = 1e-8
                       ) -> numpy.ndarray:

    # Mask the array by the MASK
    masked_array: numpy.ma.core.MaskedArray = numpy.ma.masked_invalid(x)

    # Fill the masked array with 0
    return masked_array.filled(fill_value=mask_number)
