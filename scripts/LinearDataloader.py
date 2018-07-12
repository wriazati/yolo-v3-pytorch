import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module, Linear, MSELoss, Sigmoid, ReLU, BCELoss, Tanh, RReLU, Hardtanh
from torch.optim import SGD, Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader

#######################
#   Hyperparameters   #
#######################
learning_rate = 0.01
batch_size = 32
num_iterations = 500

###################
#   Define X, Y   #
###################
class DiabetesDataset(Dataset):
	def __init__(self):
		XY = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
		self.len = XY.shape[0]
		self.X   = torch.from_numpy(XY[:, 0:-1])
		self.Y   = torch.from_numpy(XY[:, [-1]])

	def __getitem__(self, index):
		return self.X[index], self.Y[index]

	def __len__(self):
		return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


####################
#   Define Model   #
####################
class Model(Module):
	def __init__(self):
		super(Model, self).__init__()
		self.l1 = Linear(8, 6)
		self.l2 = Linear(6, 4)
		self.l3 = Linear(4, 1)
		self.sigmoid = Sigmoid()

	def forward(self, X):
		A1     = self.sigmoid(self.l1(X))
		A2     = self.sigmoid(self.l2(A1))
		Y_pred = self.sigmoid(self.l3(A2))
		return Y_pred

model = Model()

#########################
#   Define Loss & Opt   #
#########################
criterion = BCELoss(size_average=False)
optimizer = Adam(params=model.parameters())

#####################
#   Train Network   #
#####################
for epoch in range(num_iterations):
	for i, data in enumerate(train_loader, 0):
		# Get inputs
		X, Y = data

		# Wrap
		X, Y = Variable(X), Variable(Y)

		# Forward
		Y_pred = model.forward(X)

		# Compute Loss
		loss = criterion(Y_pred, Y); print(loss.data) if epoch % 100 == 0 else None

		# Zero Grads
		optimizer.zero_grad()

		# Backward
		loss.backward()

		# Update
		optimizer.step()

####################
#   Test Network   #
####################