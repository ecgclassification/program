import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from math import exp


from random import random

file_location = r'I:\Git Hub\ECG clssi\ECG code\training_and_testing.xlsx'
workbook = xlrd.open_workbook(file_location)
#input matrix

ip = np.empty((200,30))
file_location = r'training_and_testing.xlsx'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,200):
  xi= [first_sheet.cell_value(j,i) for i in range (30)]
  print xi
  ip[j,:]=(xi)
  for inpt in ds:
    [ xi ]
#isize=np.shape(xi)
#print ip

#output matrix
op = np.empty((200,3))

file_location = r'training_and_testing.xlsx'
workbook = xlrd.open_workbook(file_location)
second_sheet = workbook.sheet_by_index(1)
for j in range (0,20):
  xo= [second_sheet.cell_value(j,i) for i in range (3)]
  print xo
  op[j,:]=(xo)
#osize=np.shape(xo)
#print osize
'''
# sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid1(x):
  return sigmoid(x)* (1 - sigmoid(x))
 '
# Find the min and max values for each column

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
minmax= dataset_minmax(ip)


nip = normalize_dataset(ip,minmax)
n_inputs = len(ip[0])
n_outputs = len(op[0])
network = initialize_network(n_inputs, 10, n_outputs)
train_network(network, ip, 0.5, 20, n_outputs)
for layer in network:
	print(layer)
'''
