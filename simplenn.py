import numpy as np
from weights import weights
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ann:
	
	# Initialize network
	def __init__(self, n_hiddenlayers, n_inputs, n_hidden, n_outputs):
		self.weights = []
		self.shapes = []
		self.lower_bound = -0.7
		self.upper_bound = 0.7
		self.activation_function = lambda x : 1/(1+np.exp(-x))
		self.filename = "1in_linear.TXT"
		self.biased = 1 			# If 1 - bias nodes used; 0 - bias nodes unused
		self.biasedValue = 1 		# This is the value of all bias nodes

        # each layer has an extra node that's set to deliver a value of 1. It has NO need to have weights going in
		input_layer = np.random.uniform( low=self.lower_bound, high=self.upper_bound,size=(n_inputs + self.biased, n_hidden + self.biased))
		self.weights.append(input_layer)

		for i in range(n_hiddenlayers):
			hidden_layer = np.random.uniform( low=self.lower_bound, high=self.upper_bound,size=(n_hidden + self.biased, n_hidden + self.biased) )
			self.weights.append(hidden_layer)

		output_layer = np.random.uniform( low=self.lower_bound, high=self.upper_bound,size=(n_hidden + self.biased, n_outputs) )
		self.weights.append(output_layer)

	# Calculate neuron activation for an input
	def NNoutput(self, inputs):
		inputs = np.array(inputs, ndmin=2).T
		hidden_inputs = np.array([])
		i = 0
		for layer in self.weights:
			# Code to add a biased node on each layer outputting 1, can be changed to other values
			if self.biased == 1:
				if i == 0:
					inputs = np.append(inputs,[[self.biasedValue]])
				if i < (len(self.weights)-1):
					
					inputs[-1] = self.biasedValue
					#print("this is what the input looks like")
					#print(type(inputs))
					#print(inputs)
					#print(inputs[-1])
			#print(f"shape of layer is {layer.shape} and shape of inputs is {inputs.shape}")
			hidden_inputs = np.dot(layer.T, inputs)
			inputs = self.activation_function(hidden_inputs)
			i = i + 1

		return inputs

	# Self explanatory
	def chooseActivationFunction(self, choice):
		if choice == 1:
			self.activation_function = lambda x: 0
		elif choice == 2:
			self.activation_function = lambda x: 1/(1 + np.exp(-x)) 
		elif choice == 3:
			self.activation_function = lambda x: np.tanh(x)
		elif choice == 4:
			self.activation_function = lambda x: np.cos(x)
		elif choice == 5:
			self.activation_function = lambda x: np.exp((-x**2)/2)
	
	# Evaluate the ANN's performance using Mean Sqaured Error
	def evaluate(self):
		# Open file
		test_data_file = open(self.filename, 'r')		# IMPORTANT: The file needs to be in the same folder. 
		test_data_list = test_data_file.readlines()
		test_data_file.close

		i = 0
		error = 0

		# Go through all records in the training data set
		for record in test_data_list:
			# split the record by the ','
			all_values = record.split(',')

			# each data subset is a string with two float numbers, we split them up with the spaces
			inputOutput = test_data_list[i].split()
			
			# Scale and shift the inputs, the float is because the data store the numbers as strings
			# Check length of inputOutput list, if 3 numbers - treat first 2 as inputs (else only 1st), last as output
			if len(inputOutput) == 3:
				inputs = [float(inputOutput[0]), float(inputOutput[1])]
				targets = float(inputOutput[2])
			else:
				inputs = [float(inputOutput[0])] 
				targets = float(inputOutput[1])
			# query the network
			output = self.NNoutput(inputs)
			# calculate the distance to error
			error = ((targets-output)**2) + error
			# We need an I to count 
			i = i + 1
		# Calculate average error of neural network
		return -(error/i)	# Minus is there to make it easier for pso :)
	
	def display(self):
		test_data_file = open(self.filename, 'r')
		test_data_list = test_data_file.readlines()
		test_data_file.close

		i = 0
		targets = []
		outputs = []
		inputs = []
		inputs4Graph1 = []		# need to collect whole inputs array to plot to graph !!
		inputs4Graph2 = []
		targets4Graph = []		# same for targets

		for record in test_data_list:
			all_values = record.split(',')
			inputOutput = test_data_list[i].split()
			if len(inputOutput) == 3:
				inputs = [float(inputOutput[0]), float(inputOutput[1])]
				inputs4Graph1.append(float(inputOutput[0]))
				inputs4Graph2.append(float(inputOutput[1]))
				targets = float(inputOutput[2])
				targets4Graph.append(targets)
			else:
				inputs = float(inputOutput[0])
				inputs4Graph1.append(inputs)
				targets = float(inputOutput[1])
				targets4Graph.append(targets)
			outputs.append(self.NNoutput(inputs))
			i = i + 1
		
		results = []
		for i in range(len(outputs)):
			results.append(outputs[i]) 
		if len(inputOutput) == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter3D(inputs4Graph1, inputs4Graph2, targets4Graph, marker='^')
			ax.scatter3D(inputs4Graph1, inputs4Graph2, results, marker='o')
		else:
			plt.plot(inputs4Graph1, self.flatten(np.asarray(results)), 'b')	# Cast as np array to fix null act f-n because flatten() takes np arrays not ints
			plt.plot(inputs4Graph1, targets4Graph, 'r')
			plt.xlabel('X Axis')
			plt.ylabel('Y Axis')
		
		plt.draw()

# Flattening of np arrays
# arr = incoming array in list of matrices form
	def flatten(self,arr):
		if self.shapes:         # Clear shapes if shapes exist
			self.shapes = []

		new_arr = np.array([]) 
		#print(arr)
		for i in range(len(arr)):
			new_arr = np.append(new_arr, arr[i].flatten())
			self.shapes.append(arr[i].shape)

		return new_arr

# Bring it all back together...
# Only need to have shapes[] outside as self.shapes
	def rebuild(self, new_arr):
		restoredArray = []
		offset = 0
		for i in range(len(self.shapes)):
			s = self.shapes[i]
			n = np.prod(s)
			restoredArray.append(new_arr[offset : (offset+n)].reshape(s))
			offset += n

		return restoredArray


# SET UP TESTING ENV
'''
ann = ann(1,5,1)
ann.choseActivationFunction(3)
#(f"evaluation result is: {ann.evaluate()}")
print(f"total error is {ann.evaluate()} percent")
#print(ann.weights[1].shape)
#print(w.flatten(ann.getWeights())) # pass this to PSO
#print(ann.getWeights())
# '''