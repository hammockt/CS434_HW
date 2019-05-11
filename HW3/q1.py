import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(32*32, 100)
		self.fc1_drop = nn.Dropout(0.2)
		self.fc3 = nn.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 32*32)
		x = torch.sigmoid(self.fc1(x))
		x = self.fc1_drop(x)
		return F.log_softmax(self.fc3(x), dim=1)

def train(model, training_loader, device, optimizer, criterion, losses):
	# Set model to training mode
	model.train()
	loss = None

	# Loop over each batch from the training set
	for batch_idx, (data, target) in enumerate(training_loader):
		# Copy data to GPU if needed
		data = data.to(device)
		target = target.to(device)

		# Zero gradient buffers
		optimizer.zero_grad()

		# Pass data through the network
		output = model(data)

		# Calculate loss
		loss = criterion(output, target)

		# Backpropagate
		loss.backward()

		# Update weights
		optimizer.step()

	losses.append(loss.data.item())

def validate(model, test_loader, device, accuracy_vector):
	model.eval()
	correct = 0
	for data, target in test_loader:
		data = data.to(device)
		target = target.to(device)
		output = model(data)
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	accuracy = float(100. * correct.to(torch.float32) / len(test_loader.dataset))
	accuracy_vector.append(accuracy)

def main():
	device = None
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	print('Using PyTorch version:', torch.__version__, ' Device:', device)

	################
	# Data Loading #
	################

	batch_size = 32

	#transform = transforms.Compose([
	#	transforms.ToTensor(),
	#	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
	#])
	transform = transforms.Compose([
	    transforms.Grayscale(num_output_channels=1),
	    transforms.ToTensor()
	])

	dataset = datasets.CIFAR10('./data',
	                           train=True,
	                           download=True,
	                           transform=transform)

	test_dataset = datasets.CIFAR10("./data",
	                                train=False,
	                                transform=transform)

	training_pivot = int(len(dataset) * 0.80)
	training_range = range(training_pivot)
	validation_range = range(training_pivot, len(dataset))

	training_dataset = torch.utils.data.Subset(dataset, training_range)
	validation_dataset = torch.utils.data.Subset(dataset, validation_range)

	training_loader = torch.utils.data.DataLoader(dataset=training_dataset,
	                                              batch_size=batch_size,
	                                              shuffle=True)

	validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
	                                                batch_size=batch_size,
	                                                shuffle=False)

	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
	                                          batch_size=batch_size,
	                                          shuffle=False)

	########################################
	# Multi-Layer Perceptron Network Class #
	########################################

	model = Net().to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
	criterion = nn.CrossEntropyLoss()

	#################
	# Learning Code #
	#################

	EPOCHS = 5

	training_losses = []
	validation_accuracies = []
	test_accuracy = []
	for epoch in range(1, EPOCHS + 1):
		train(model, training_loader, device, optimizer, criterion, training_losses)
		validate(model, validation_loader, device, validation_accuracies)
	validate(model, test_loader, device, test_accuracy)

	#plot the validation accuracy
	fig, axes1 = plt.subplots()
	axes2 = axes1.twinx()
	axes1.plot(range(1, EPOCHS + 1), training_losses, c='b', label="training_losses")
	axes2.plot(range(1, EPOCHS + 1), validation_accuracies, c='r', label="validation_accuracies")
	plt.title("Effects of Epochs on Testing loss and Validation Accuracy")
	plt.xlabel("Epochs")
	axes1.set_ylabel("Training Loss")
	axes2.set_ylabel("Validation Accuracies")
	plt.legend(loc="best")
	plt.show()

	print(f"Testing accuaracy: {validation_accuracies[0]}")

main()
