import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.tensor(7))

# print(torch.tensor([7,8]))

# print(torch.tensor([[7,8],
#                     [3,5]]))

# print(torch.tensor([[[1,2],
#                      [3,4]],

#                     [[5,6],
#                      [7,8]]]))


# random_tensor = torch.rand(size=(2, 3, 4))
# ones_tensor = torch.ones(size=(10,10))
# zeros_tensor = torch.zeros(size=(1,12))

# print(random_tensor);
# print(ones_tensor);
# print(zeros_tensor)


# one_to_ten = torch.arange(start=1, end=11, step=1);
# ten_zeros = torch.zeros_like(one_to_ten)
# ten_ones = torch.ones_like(ten_zeros)
# print(one_to_ten);
# print(ten_zeros)
# print(ten_ones);

# float_32_tensor = torch.tensor(
#     [3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False
# )

# print(float_32_tensor)

# float_16_tensor = float_32_tensor.type(torch.float16);

# print(float_16_tensor.dtype)

# tensor = torch.tensor([[1,2,3],
#                       [4,5,6]])

# tensor2 = torch.tensor(tensor);
# tensor3 = torch.matmul(tensor,tensor2.T)
# print(tensor3)

# print(torch.mean(tensor3.type(torch.float32)))

# x = torch.arange(1,10)
# x_shaped = x.reshape(1,3,3)
# print(x_shaped)

# print(x_shaped[0,:,1])
