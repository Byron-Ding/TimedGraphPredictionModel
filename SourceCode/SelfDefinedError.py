import torch


class MatrixCompatabilityError(Exception):
    def __init__(self, message: str,
                 command: str,
                 matrix_size_list_left: list[torch.Tensor],
                 matrix_size_list_right: list[torch.Tensor]):
        super().__init__(message)
        self.message = message
        self.command = command
        self.matrix_list_left: list[torch.Tensor] = matrix_size_list_left
        self.matrix_list_right: list[torch.Tensor] = matrix_size_list_right

    def __str__(self):
        output = "For the command: " + self.command + "\n" \
                 + self.message + "\n" + \
                 "The size of the left matrix size is: " + str(self.matrix_list_left) + "\n" + \
                 "The size of the right matrix size is: " + str(self.matrix_list_right) + "\n"
        return output
