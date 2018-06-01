import xlrd
import numpy as np
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
   import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')

import xlrd
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import messagebox
from math import *

#from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure



# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

 

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

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
###########################################################################################################################################
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	conmat= list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
		confusion_matric= confusion_matrix(actual,predicted)
		conmat.append(confusion_matric)
		#conmat.tolist()
		print (confusion_matric)

	return (scores,conmat)


    

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
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
			
			
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


# load and prepare data
filename = r'datadctip.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

# normalize input variables

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 2
l_rate = 0.3
n_epoch = 500
n_hidden = 10
scores,conmat = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores) 
s=(sum(scores)/float(len(scores)))                                           # individual acc
print('Mean Accuracy: %.3f%%' % s)       #avg acc
print( conmat[0])                                                       # confusion matrix1
print( conmat[1])                                                       # confusion matrix2


###########################gui

a=np.array(conmat[0])
b=np.array(conmat[1])

m = a[0][0] + b[0][0]
n = a[1][1] + b[1][1]
p = a[2][2] + b[2][2]

print(m)
print(n)
print(p)

 
if (m > n) and (m > p):
   largest = "The patient is normal"
elif (n > m) and (n > p):
   largest = " The patient is diogose with PVC "
else: 
   largest = " The patient is diogose with FUSSION "
 
print(largest)





def write_slogan():
    print("the patient is normal")
    
def hello():
   msg4 = tk.Label(root, compound =tk.LEFT, padx = 10, text='%.3f%%' % s)
   msg4.config(bg='white', font=('times', 20, 'italic'))
   msg4.place(x =680 ,y = 20)
   
   msg5 = tk.Label(root, compound =tk.LEFT, padx = 10, text=conmat[0])
   msg5.config(bg='white', font=('times', 15, 'italic'))
   msg5.place(x =620 ,y = 140)
   
   msg6 = tk.Label(root, compound =tk.LEFT, padx = 10, text=conmat[1])
   msg6.config(bg='white', font=('times', 15, 'italic'))
   msg6.place(x =620 ,y = 260)
   
   messagebox.showinfo("RESULT", largest) 
       
def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate   

               
                    
def PVC():
    img = PhotoImage(file="PVC.gif")
    canvas.create_image(5,-5, anchor=NW, image=img)
    root.mainloop()

def NORMAL():
   img3 = PhotoImage(file="normal.gif")
   canvas.create_image(1,-5, anchor=NW, image=img3)
   canvas.pack(expand = YES, fill = BOTH)
   root.mainloop()
   
def FUSION():
   img4 = PhotoImage(file="fusion.gif")
   canvas.create_image(30,-5, anchor=NW, image=img4)
   root.mainloop()

root = tk.Tk()
frame = tk.Frame(root)  
frame.pack()


root.wm_title("ECG SIGNAL CLASSIFIER")

canvas_width = 800
canvas_height =480



canvas = Canvas(root, width=canvas_width, height=canvas_height,bg = 'white')
canvas.pack(expand = YES, fill = BOTH)
        

img2 = PhotoImage(file="viit.gif")
canvas.create_image(50,280, anchor=NW, image=img2)

img6 = PhotoImage(file="ecg.gif")
canvas.create_image(30,10, anchor=NW, image=img6)

img7 = PhotoImage(file="VAT.gif")
canvas.create_image(530,220, anchor=NW, image=img7)

img8 = PhotoImage(file="VAT.gif")
canvas.create_image(530,100, anchor=NW, image=img8)


def motion(event):
        print("Mouse position: (%s %s)" % (event.x, event.y))
        return 
 
''' 

msg = tk.Message(root,justify =tk.LEFT, text = explanation)
msg.config(bg='lightblue', font=('times', 16, 'italic'))
msg.place(x = 240,y = 550)
msg.bind('<Motion>',motion)
'''
#tk.Label(root,justify=LEFT,compound = LEFT, text=txt, fg = "darkblue", bg = "lightblue", font = "Verdana 80 italic").place(x=200,y=150)
        
def donothing():
   x = 0
 
#root = Tk()
''' 
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=PVC)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=_quit)
menubar.add_cascade(label="File", menu=filemenu)
 
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=donothing)
menubar.add_cascade(label="Help", menu=helpmenu)
 
root.config(menu=menubar)
'''                 
                 
explanation = """Project Name:
 Multi-Class ECG signal Clasification
 shraddha Kapse 411028
 Sunny C.J Francis 413017
 Sanket Naik 412018"""
 
msg = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text=explanation)
msg.config(bg='white', font=('times', 12, 'italic'))
msg.place(x = 200,y = 300) 


msg2 = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text='Accuracy : ')
msg2.config(bg='white', font=('times', 20, 'italic'))
msg2.place(x = 550,y = 20) 

msg3 = tk.Label(root, 
              compound =tk.LEFT,
              padx = 10, 
              text='Confusion Matrix:')
msg3.config(bg='white', font=('times', 16, 'italic'))
msg3.place(x = 550,y = 70)

#x = 96

                        

'''
x = range(9)

x = reshape(x,(3,3)) 
'''
'''
logo = tk.PhotoImage(file="viit.gif")
w1 = tk.Label(root, image=logo).pack(side="left")
'''
'''
def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)
canvas.mpl_connect('key_press_event', on_key_event)
'''    
     
                           
button = tk.Button(master=root, text='CLEAR',fg="red", command=root.quit)
button.place(x = 320,y = 430) 

button2 = tk.Button(master=root, padx = 10, text='RESULT', command=hello)
button2.place(x = 560,y = 380) 

button3 = tk.Button(master=root, padx = 10, text='PVC', command=PVC)
button3.place(x = 480,y = 430)

button4 = tk.Button(master=root, padx = 10, text='NORMAL', command=NORMAL)
button4.place(x = 560,y = 430)

button5 = tk.Button(master=root, padx = 10, text='FUSION', command=FUSION)
button5.place(x = 660,y = 430)


mainloop()      


