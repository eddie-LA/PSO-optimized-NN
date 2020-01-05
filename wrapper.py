import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pso2

''' Goal: Create a method / class that can call pso2 iteratively with different params, 
    save the values for the graphs, builds a graph after an average of N runs,
    saves the graph as a file with the params as filename.

    '''


''' pso_wrapper():
    Takes: nothing, Returns: nothing
    This method takes numpy ranges of type
'''
class pso_wrapper():
    def __init__(self):
        # How many times to run and average over?
        self.runs = 10

        # PSO params
        self.S = 20             # Swarm size
        self.limit = 2          # Limits of the search space
        self.step_size = 0.6
        self.inf_size = 6
        self.max_iter = [100]   # Need a positive whole value

        self.inertia_factor = [0.7, 0.9, 1, 1.1]
        self.self_confidence = 1.6
        self.informant_confidence = 1.8
        self.swarm_confidence = [1.3, 1.5, 2, 2.1]

        # ANN params - input and output nodes are decided automatically
        self.number_hlayers = [1, 2, 3]
        self.hidden_nodes = 6
        self.filenames = ["1in_cubic.TXT","1in_linear.TXT","1in_sine.TXT","1in_tanh.TXT","2in_xor.TXT", "2in_complex.TXT"]     # Filename to run PSO NN on - default: "1in_linear.TXT"
        self.act_fn = [2, 3, 4, 5] # 1 - null, 2 - sigmoid, 3 - tanh, 4 - cos, 5 - exp(x^2/2)
    

    def iterate(self):
        # You can be more specific about some params here...
        #filename = self.filenames[3]        # set to desired filename from list on top
        graph_array = []
        results = []
        counter = 0     # count how many graphs we need
        
        for filename in self.filenames:
            for act_func in self.act_fn:
                for iterations in self.max_iter:
                    for inertia in self.inertia_factor:
                        for swarm_conf in self.swarm_confidence:
                            for hlayers in self.number_hlayers:

                                for runs in range(self.runs):
                                    graph_array.append( np.asfarray( pso2.run(filename, act_func, iterations, inertia, swarm_conf, hlayers) ) )
                                
                                # Results from all 10 runs done, average and append them to final results
                                results = sum(graph_array) / self.runs
                                #print(results)

                                # Create our folder if it doesn't exist
                                if not os.path.exists('Graphs'):
                                    os.makedirs('Graphs')
                                
                                # Graph bestNN from current PSO
                                plt.savefig(f".\Graphs\{filename}_act-fn-{act_func}_iterations-{iterations}_inertia-{inertia}_swarm-conf-{swarm_conf}_hlayers-{hlayers}-BEST-NN-RESULTS.png")
                                plt.close()

                                # Graph!
                                fig, ax = plt.subplots()
                                ax.plot(np.arange(0, iterations), results[0]) # first part of numpy array is bfit, this graphs bfit
                                ax.plot(np.arange(0, iterations), results[1]) # second element of numpy array is avgfit, this graphs avgfit
                                ax.legend(['bfit', 'avgfit'], loc='upper left')
                                ax.set(xlabel='Iterations (num)', ylabel='fitness value', title='Convergence graph')
                                ax.grid()   # Show the grid                     
                                
                                # Save graph to Graphs folder
                                plt.savefig(f".\Graphs\{filename}_act-fn-{act_func}_iterations-{iterations}_inertia-{inertia}_swarm-conf-{swarm_conf}_hlayers-{hlayers}.png")
                                plt.close()         # clear figure, otherwise we get many on top of each other!
                                plt.close()         # prevent possible memory leaks
                                graph_array = []    # clear graph array for use in following loop

            

# Let's rock!
w = pso_wrapper()
w.iterate()