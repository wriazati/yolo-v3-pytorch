from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module, Linear, MSELoss
from torch.optim import SGD

#######################
#   Hyperparameters   #
#######################

###################
#   Define X, Y   #
###################

####################
#   Define Model   #
####################

#########################
#   Define Loss & Opt   #
#########################

#####################
#   Train Network   #
#####################

####################
#   Test Network   #
####################



# -------------------------------------



###################
#   Define X, Y   #
###################
x_data = Tensor([[1.0], [2.0], [3.0]])
y_data = Tensor([[2.0], [4.0], [6.0]])

####################
#   Define Model   #
####################
class Model(Module):

	def __init__(self):
		super(Model, self).__init__()

		# One in, one out
		self.linear = Linear(1, 1)

	def forward(self, x):
		"""
		In forward we accept Variable of input data and we return a Variable of output data
		:param input:
		:return:
		"""
		y_pred = self.linear(x)
		return y_pred

model = Model()


#########################
#   Define Loss & Opt   #
#########################
criterion = MSELoss(size_average=False)
optimizer = SGD(model.parameters(), lr=0.01)


#####################
#   Train Network   #
#####################
for epoch in range(500):
	# Forward pass
	y_pred = model.forward(x_data)
	# Compute loss
	loss = criterion(y_pred, y_data) ; print(epoch, loss.data) if epoch % 100 == 0 else 1
	# Zero out grads
	optimizer.zero_grad()
	# Back pass
	loss.backward()
	# Update weights
	optimizer.step()


####################
#   Test Network   #
####################
my_x = Variable(Tensor([[4.0], [5.0]]))
print("Prediction: ", model.forward(my_x).data)
print("Expected  : [8.0], [10.0]")
