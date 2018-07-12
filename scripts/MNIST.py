import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module, Linear, MSELoss, Sigmoid, ReLU, BCELoss, Tanh, RReLU, Hardtanh, CrossEntropyLoss, Conv2d, \
	MaxPool2d, Softmax
from torch.optim import SGD, Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F

#######################
#   Hyperparameters   #
#######################
learning_rate = 0.01
batch_size = 64
num_iterations = 2


###################
#   Define X, Y   #
###################
train_loader = DataLoader(
	batch_size=batch_size,
	shuffle=True,
	dataset=datasets.MNIST(
		'data',
		train=True,
		download=True,
		transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])),
	)

test_loader = DataLoader(
	batch_size=batch_size,
	shuffle=True,
	dataset=datasets.MNIST(
		'data',
		train=False,
		download=True,
		transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])),
	)



####################
#   Define Model   #
####################
class Model(Module):
	def __init__(self):
		super(Model, self).__init__()                                       # 28x28x1
		self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)   # 24x24x6
		self.mp1   = MaxPool2d(kernel_size=2)                               # 12x12x6
		self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=3)  # 10x10x16
		self.mp2   = MaxPool2d(kernel_size=2)                               # 5x5x16
		self.fc1   = Linear(400, 120)
		self.fc2   = Linear(120, 84)

	def forward(self, X):
		X = F.relu(self.conv1(X))
		X = F.relu(self.mp1(X)  )
		X = F.relu(self.conv2(X))
		X = F.relu(self.mp2(X)  )
		X = X.view(-1, 400)
		X = F.relu(self.fc1(X)  )
		X = F.relu(self.fc2(X)  )
		return F.log_softmax(X, dim=1)

model = Model()


#########################
#   Define Loss & Opt   #
#########################
criterion = CrossEntropyLoss(size_average=True)
optimizer = Adam(params=model.parameters())

#####################
#   Train Network   #
#####################
def train(model, train_loader, optimizer, epoch):
	model.train()

	for batch_index, (X, Y) in enumerate(train_loader):
		# Forward
		Y_pred = model.forward(X)
		# Compute Loss
		loss = criterion(Y_pred, Y)
		# Zero Grads
		optimizer.zero_grad()
		# Backward
		loss.backward()
		# Update
		optimizer.step()
		# Print
		if batch_index % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_index * len(X), len(train_loader.dataset),
				       100. * batch_index / len(train_loader), loss.item()))

####################
#   Test Network   #
####################
def test(model, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for X_test, Y_test in test_loader:
			Y_pred = model(X_test)
			test_loss += F.nll_loss(Y_pred, Y_test, size_average=False).item() # sum up batch loss
			pred = Y_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(Y_test.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	    test_loss, correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))



##############
#   Runner   #
##############

for epoch in range(num_iterations):
	train(model, train_loader, optimizer, epoch)
	test(model, test_loader)