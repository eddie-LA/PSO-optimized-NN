# PSO-optimized ANN
A customizable, modular Artificial Neural Network in Python optimized by a Particle Swarm Optimization algorithm. In its current form it is used to approximate functons from a text file with input/output pairs, located in the Data folder. Each particle in the PSO is an ANN - its list of weights, the particle's position.

The code includes a wrapper class for bulk exection and testing. Upon execution, it creates a folder called Graphs inside its current folder and creates graphs of whatever the current run and its parameters are with a descriptive name.

The graphs included are of 2 types - a fitness graph of the particle swarm, showing how the fitness of the swarm improves with time and a resultant function (blue) graphed vs actual function (red). The idea would be that in a real world scenario one would average out the N number of runs returning blue to get the best possible result, closest to the red actual value.

To run - simply keep the folder structure, run (or modify and run if you wish) the wrapper with your favourite Python interpreter.
Requires Python 3. Written in Python 3.8 but should run in 3.7. Python 2 will give you an error if you try to run it with that, due to the use of F strings.

## Customizable parameters
All parameters are customizable if you look thorugh the code with minor modifications here and there. Not all of them are very relevant for running in default state, so only the relevant ones have been exposed in the wrapper.

### Wrapper parameters
The number of runs to do

### PSO parameters
Swarm size - how many ANNs to create 
Search space limits - this defines the boundaries of the particle positions 
Step size - a factor to scale velocity by, used to make particles overall faster or slower - one of the main convergence controls
Informant pool size - how many particles will each particle be informed by
Max iterations - amount of iterations to do in the PSO - time/accuracy tradeoff 
Particle velocity factors - inertia, self confidence, informant confidence, swarm confidence

### ANN parameters
The input/output pairs - keep within the same format!
Activation function choices - Null, Sigmoid, tanh (Hyperbolic tangent), cosine and complex (exponential).
Number of hidden layers - any positive integer
Number of hidden neurons - any positive integer

* Note: Some graphs will turn out looking bad. This is due to iteration over activation functions for problems that require a specific one (e.g. an activation funcion with only positive values so negatives can't be handled in the neurons, where the output needs to be negative in places). This was needed to do and point out in the report and is working well.
