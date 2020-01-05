import sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import copy
from simplenn import ann

# PSO Hyperparameters
S = 30           # Swarm size
limit = 2        # Limits of the search space
step_size = 0.4
inf_size = 5
max_iter = 100   # Need a positive whole value
debug = 0        # Tracks the first particle around the search space

inertia_factor = 0.8
self_confidence = 0.75
informant_confidence = 1.8
swarm_confidence = 2.1

# ANN variables
number_hlayers = 2
input_nodes = 2
hidden_nodes = 10
output_nodes = 1
filename = "2in_xor.TXT"     # Filename to run PSO NN on - default: "1in_linear.TXT"
act_fn = 4


''' class particle
Particle made of:
    pos 
    fitness value
    velocity/displacement - used to compute next pos
    bpos - best position memory
    bfit - previous best fitness

Informants provide their:
    bpos
    bfit 

to the particle. Informants include particle itself in SPSO but not necessary.
'''
class Particle:
    def __init__(self, nn):
        self.nn = nn
        self.weights = nn.flatten(nn.weights)
        self.pos = self.weights              # num of dimensions, one d for every weight or bias, [-limit:limit)
        self.bpos = self.pos.copy()
        self.fit = 0                                          # fitness, up to David to implement fatness f-n
        self.bfit = 0                                           # best fitness
        self.vel = (np.random.random([len(self.weights)]))             # velocity of particle, [-1:1]
        self.informants = []                                     # setting an empty list for the informants

    def updateVelocity(self,prevb_self_pos, prevb_inf_pos, prevb_all_pos):
        gp = random.random()*self_confidence           # retain random amount of self_confidence (personal best)
        gi = random.random()*informant_confidence      # same from informant_confidence (informants' best)
        gg = random.random()*swarm_confidence          # same for swarm_confidence (global best)
        
        # Vel update equation
        self.vel = inertia_factor*self.vel + gp*(prevb_self_pos - self.pos) + gi*(prevb_inf_pos - self.pos) + gg*(prevb_all_pos - self.pos)
    
    def move(self):
        self.pos = self.pos + step_size*self.vel      # move particle (step_size - factor to move by)
        
        np.clip(self.pos, -limit, limit)        # Fast! Particle confinement

        # Particle confinement
        '''for i in range(self.pos.size):          # check if out of bounds
            if self.pos[i] > limit:
                self.pos[i] = limit
                self.vel[i]*=0            # reflection of particle off wall + slowing
            if self.pos[i] < -limit:
                self.pos[i] = -limit
                self.vel[i]*=0  '''           # reflection of particle off wall (negative) + slowing

    def addInformant(self, new):
        self.informants.append(new)


# TESTING 1 2 3, Fatness f-n
def test_fitness(pos):
    #fitness = -(pos[0]**2 + pos[1]**2)
    fitness = - (pos[0]**2 - 1)**2 - (pos[0]**2*pos[1]-pos[0]-1)**2+200
    return fitness

def coord_2_magnitude(vel_vector):
    return np.sqrt( np.sum( np.square(vel_vector) ) )

''' class pso

init_swarm()
    pick random pos in search space
    compute fitness
    bpos and bfit = current pos and fit
    pick random vel

iterate()
    updateVel() - compute new vel by taking:
        current pos
        current vel
        bpos
        bpos of all informants

    move()
        apply new vel to every particle

    limit()
        apply a confinement method to ensure particle is in search space
        compute new fitness
            (OR let them fly method - no confinement but also no fitness reevaluation)
    
    fitnessCompute()
        if p.fit is better than p.bfit -> bfit = fit.copy(), bpos = pos.copy()

 STOP CRITERIA:
    when fitness of optimum point is known (optimum_point_fit)
        when abs( optimum_point_fit - bfit ) < some predefined min error value
    
    OR
    
    reached max_iter
'''
class pso:
    def __init__(self):
        # Initialize particles and overall swarm bfit
        self.particles = []
        self.a = []
        self.bfit = np.asfarray(-limit**2)       # Value at the limit of our search space, we aren't gonna get a result worse than this
        self.bestNN = None

        # Graph Helper variables
        self.graph_avgfit = []
        self.graph_bfit = []


        for i in range(S):
            # Create NN variables with selected criteria
            nn = ann(number_hlayers,input_nodes,hidden_nodes,output_nodes)
            nn.chooseActivationFunction(act_fn)
            nn.filename = filename      
            self.a.append(nn)

            # Create Particle objects and feed them the NN information
            p = Particle( self.a[i] )            
            self.particles.append(p)
            p.bpos = p.pos.copy()
            p.fit = np.copy(self.a[i].evaluate())
            #print(f"p.fit is {p.fit}")
            p.bfit = np.copy(p.fit)

        # Add informants to particles
        for p in self.particles:
            p.addInformant(p) # add self to informants, respecting SPSO
            for i in range(inf_size - 1):
                p.addInformant(self.particles[random.randrange(S)])
    
    def updateFitness(self, p):
        if(p.fit > p.bfit):
            p.bfit = p.fit.copy()
            p.bpos = p.pos.copy()
    
    def avgFitness(self):
        avg = 0
        for p in self.particles:
            avg += p.bfit
        return avg/S

    def optimize(self):

        for i in range(max_iter):
            if debug == 1:          # Track particle 0
                print(f"Tracked particle is {self.particles[0].pos} and its fitness is {self.particles[0].fit}")
                print(f"Its velocity is {coord_2_magnitude(self.particles[0].vel)}")

            for p in range(len(self.particles)):
                #if p.bfit > -0.1: <- Old code for premature convergence
                    #return print(f"Successfully converged at {p.pos} with a bfit of {p.fit}, avg bfit of swarm is {self.avgFitness()} after {i} iterations.")
                
                ''' 1. Find and keep track of the best fitness of every individual particle
                    2. Look for overall best fitness from swarm (self.bfit)
                '''
                prevb_self_pos = self.particles[p].bpos.copy()

                if self.particles[p].bfit > self.bfit:
                    self.bfit = self.particles[p].bfit
                    self.bestNN = copy.deepcopy(self.particles[p].nn)
                    #print(f"bfit is {self.bfit} and evaluated bfit was {self.bestNN.evaluate()}")

                ''' 3. Find best fitness in the whole swarm in this time step:
                    - Initialize to first particle
                    - Loop through rest of swarm, comparing best fitness of all particles
                '''
                prevb_swarm_fit = self.particles[0].bfit
                prevb_swarm_pos = self.particles[0].bpos
                for n in self.particles[1:]:
                    if  n.bfit > prevb_swarm_fit: 
                        prevb_swarm_fit = n.bfit
                        prevb_swarm_pos = n.bpos

                ''' 4. Same as above...
                    - Look into informants and find the best pos/fit from all informants
                    - Use current particle's best position, best informant position and best swarm position to updateVelocity()
                ''' 
                prevb_inf_fit = self.particles[p].informants[0].bfit
                prevb_inf_pos = self.particles[p].informants[0].bpos
                for m in self.particles[p].informants[1:]:          # Take prev overall fittest pos from informants of p
                    if m.bfit > prevb_inf_fit: 
                        prevb_inf_fit = m.bfit
                        prevb_inf_pos = m.bpos
                    
                self.particles[p].updateVelocity(prevb_self_pos, prevb_inf_pos, prevb_swarm_pos)

                #print(coord_2_magnitude(self.particles[p].vel))

                self.particles[p].move()    # Move particle, applying limits ot dimensions

                self.a[p].weights = self.a[p].rebuild( self.particles[p].pos )   # Rebuild list of weight matrices after PSO computation
                
                self.particles[p].fit = self.a[p].evaluate()     # Compute fitness

                self.updateFitness(self.particles[p])       # Update bfit to p.fit
            
            # Build up a fitness graph
            self.graph_avgfit.append(self.avgFitness().item(0))  # item() is necessary because the object is a numpy array
            self.graph_bfit.append(self.bfit.item(0))

        #print(f"bfit is {self.bfit} and evaluated bfit was {self.bestNN.evaluate()}")
        self.bestNN.display()
        #return print(f"Converged with a bfit of {self.bfit}, avg bfit of swarm is {self.avgFitness()} after {max_iter} iterations.")
        return [self.graph_bfit, self.graph_avgfit]
             
def run(in_filename, in_act_fn, in_max_iter, in_inertia_factor, in_swarm_confidence, in_number_hlayers):
    ''' The global vars are our defaults. However, if you want to run PSO from outside
    (most likely using wrapper) you can modify this method however you want. Right now
    you can modify the filename, activation function, some PSO params and one ANN param.
    They are named with a prefix *in_*variable_name (no stars in the name!).
        To add/change the hyperparams from outside, just add them to the function 
    declaration and also below, in the same way as the current ones! :)
    
    1st line: filename and activation f-n
    2nd line: PSO params
    3rd line: ANN params 
    '''
    global filename, act_fn
    global max_iter, inertia_factor, swarm_confidence
    global number_hlayers, input_nodes
    
    filename = in_filename
    act_fn = in_act_fn
    
    # Input nodes depend on the first character of the filename!
    input_nodes = int(in_filename[0])

    # PSO Hyperparameters
    max_iter = in_max_iter
    inertia_factor = in_inertia_factor
    swarm_confidence = in_swarm_confidence

    # ANN hyperparameters
    number_hlayers = in_number_hlayers


    #print("PSO start...")
    x = pso()
    # Graph Helper variables - we need to return these in order to graph the PSO from outside...
    #graph_avgfit = x.graph_avgfit
    #graph_bfit = x.graph_bfit

    #fig, ax = plt.subplots()
    #ax.plot(np.arange(0, max_iter), x.graph_bfit)
    #ax.plot(np.arange(0, max_iter), x.graph_avgfit)
    #ax.legend(['bfit', 'avgfit'], loc='upper left')

    #ax.set(xlabel='Iterations (num)', ylabel='fitness value', title='Convergence graph')
    #ax.grid()

    #plt.show()

    return pso.optimize(x)
