from torch import Tensor
import torch
from torch.autograd import Variable
from torch.nn import MSELoss

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor(x_data),  requires_grad=True)

MSELoss

print(w)
