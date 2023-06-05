import torch

x = torch.randn(3, requires_grad=True)
print(x)

# when we do an operation the pytorch will create a computational graph
y = x + 2
print(y)  # tensor([2.6951, 1.6074, 1.0567], grad_fn=<AddBackward0>)

z = y * y * 2
print(z)  # tensor([ 8.1019,  0.3584, 11.2910], grad_fn=<MulBackward0>)
z = z.mean()
print(z)  # tensor(2.1784, grad_fn=<MeanBackward0>)

# To compute gradients now:
z.backward()  # computes dz/dx
print(x.grad)  # x.grad stores computed gradients from the Jacobian matrix

# if z is not scalar and rather a vector, then we need to use arguments inside z.backward()
z = y * y * 2

# To compute gradients now:
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)  # computes dz/dx
print(x.grad)

# Prevent tracking gradient history

# option 1
# x.requires_grad_(False)

# option 2
# x.detach()

# option 3
# with torch.no_grad():
#   y = x + 2

# example

# Here gradiends would be accumulated
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)

# but if we don't want to accumulate the gradients we use
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
