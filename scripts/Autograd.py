
import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)

# y was created as a result of an operation, so it has a grad_fn.
y = x + 2
print(f"Y: {y}")
print(f"Y grad_fn: {y.grad_fn}")


# More ops on Y
z = y * y * 3
out = z.mean()
print("Z, mean", z, out)


# Requires grad
print("Requires grad")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)























