import torch
import numpy as np

x = torch.empty(3)  # initializing tensor with 3 dimension
print(x)

y = torch.empty(2, 2) # 2-d matrix
print(y)

# random initilization
print(torch.rand(2, 2))

# initialize with 0's
print(torch.zeros(2, 2))

# initialize with 1's
z = torch.ones(2, 2)
print(z)
print(z.dtype)

# assigning with custom data size
z = torch.ones(2, 2, dtype=torch.float16)
print(z.dtype)

# checking size
print(z.size())

# tensor from list
x = torch.tensor([2.4, 5.2])
print(x)

# tensor element-wise addition, subtraction, multiplication, division
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)

print(x+y)
print(torch.add(x, y))
print(torch.sub(x, y))
print(torch.mul(x, y))
print(torch.div(x, y))

# in-place addition, substraction
print(y.add_(x))
print(x-y)

# numpy array-like numpy array
print(x[:, 0])

# to get actual value of an element
print(x[0, 1].item())

# changing the shape of tensor
x = torch.rand(4, 4)
print(x.view(16))

# to automatically determine the remaining dimension, we use -1
print(x.view(-1, 8))

# tensor to numpy conversion
print(x.numpy())

# numpy to tensor conversion
a = np.ones(5)
print(torch.from_numpy(a))
# can also provide data type
print(torch.from_numpy(a))

# creating tensor on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)

    y = torch.ones(5)
    y = y.to(device)
    # Note: we can not convert GPU tensor to numpy
    # So we will then need to send to CPU first
    z = x + y
    print(z)
    z = z.to("cpu")
    print(z.numpy())
