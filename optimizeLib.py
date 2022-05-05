#Strucutre prediction software 
#A general library that contains method used in multiple different files.
#Got an atom class.
#Potential functions

#Import needed modules
import numpy as np
from itertools import combinations

#Global sigma and epilson variables.
#Are global so can be used in generate the ground configs
sigma = 1
eps = 1

class atom:
    def __init__(self, typeA,ind):
        self.indx = ind
        self.typeAtom = typeA
        if typeA == 'A':
            self.eps = sigma
            self.sig = eps
        else:
            #Atom of type B
            self.eps = sigma
            self.sig = eps

def dist(index1, index2, posList):
    #Input the indices of the atoms in the global position list
    #Output: The distance of the atoms
    dX = posList[index1] - posList[index2]
    #dY = posList[3*index1 + 1] - posList[3*index2 + 1]
    #dZ = posList[3*index1 + 2] - posList[3*index2 + 2]
    #For 1D just use the index itself
    return abs(dX)

def LJAtom(atom1, atom2, dis):
    #In put 2 atoms and the distance between them
    #Return the energy given by the lennard jones potential
    #For AA and BB atoms uses the modified potential
    #For AB used the regular lennard jones. Bascially like an ionic crystal
    
    if atom1.typeAtom == atom2.typeAtom:
        #The same atom so interactions are repulsive
        return LJMod(dis)
    else:
        #Atoms are different so use a typical lennard jones potential
        return LJ(dis)

def LJ(r):
    #Define the Lennard jones potential
    #Input the LJ potential parameters and the distance between the objects
    #The min is at: r = 2^(1/6) sigma
    #If r = 0 throw +infinity as the value
    if r == 0:
        return np.inf
    else:
        val = sigma / r
        return 4*eps* ( (val**12) - (val**6) )

def LJMod(r):
    #Define the Modified Lennard jones potential
    #Input the LJ potential parameters and the distance between the objects
    #Used for A-A and B-B interactions
    #If r = 0 throw +infinity as the value
    if r == 0:
        return np.inf
    else:
        val = sigma / r
        return 4*eps* ( (val**12) + (val**6) )

def energySurface(atomPos):
    #Input para: the global position list of all the atoms
    #Output: the total energy of all the atoms
    
    initalEnergy = 0
    #Locally make an atomList
    atomList = []
    typeAtom = True #Goes A,B,A,...
    for i in range(len(atomPos)):
        #Add the atom
        if typeAtom:
            atomList.append( atom('A', i) ) #Append Atom A
        else:
            atomList.append( atom('B', i) ) #Append Atom B
        typeAtom = not(typeAtom)
    #Go through all pairs to evaluate
    for x in combinations(atomList, 2):
        #print( dist(x[0].indx, x[1].indx) ) # Prints the distance between each 2-combo of atom 
        initalEnergy += LJAtom(x[0], x[1], dist(x[0].indx, x[1].indx, atomPos) )
    
    return initalEnergy

def discretised(atomList, disc):
    #Input: The position list and rounding factor
    #Output: Round the positions into different bins
    #Output is int type if the input is an integer
    #This uses encoding 1 and rounds the position array to local bins
    discretePos = []
    for i in atomList:
        try:
            if disc.is_integer():
                discretePos.append(int(disc * round(i/disc)))
            else:
                discretePos.append(disc * round(i/disc))
        except:
            discretePos.append(disc * round(i/disc))
    
    return discretePos

#Code for the bottom two algorithms helped by
#https://kenndanielso.github.io/mlrefined/blog_posts/
#5_Zero_order_methods/5_5_Coordinate_search_and_descent.html

# random search function
def random_search(alpha_choice,max_its,w,num_samples):
    # run random search. Uses encoding 1 
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):        
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
            
        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(energySurface(w))
        
        # construct set of random unit directions
        directions = np.random.randn(num_samples,np.size(w))
        norms = np.sqrt(np.sum(directions*directions,axis = 1))[:,np.newaxis]
        directions = directions/norms   
        
        ### pick best descent direction
        # compute all new candidate points
        w_candidates = w + alpha*directions
        
        # evaluate all candidates
        evals = np.array([energySurface(w_val) for w_val in w_candidates])

        # if we find a real descent direction take the step in its direction
        ind = np.argmin(evals)
        if energySurface(w_candidates[ind]) < energySurface(w):
            # pluck out best descent direction
            d = directions[ind,:]
        
            # take step
            w = w + alpha*d
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(energySurface(w))
    return weight_history,cost_history

#Zero order coordinate search
def coordinate_search(alpha_choice,max_its,w):
    # construct set of all coordinate directions
    #Discretised encoding 1
    directions_plus = np.eye(np.size(w),np.size(w),dtype=int) #Identity of size W the positions
    directions_minus = - np.eye(np.size(w),np.size(w),dtype=int)
    directions = np.concatenate((directions_plus,directions_minus),axis=0)
    
    #Make sure the input is discretized
    w = discretised(w, alpha_choice)
    
    # run coordinate search
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):        
        #Could use a dimishing step rule
        alpha = alpha_choice
            
        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(energySurface(w))
        
        ### pick best descent direction
        # compute all new candidate points
        w_candidates = w + alpha*directions
        
        # evaluate all candidates
        evals = np.array([energySurface(w_val) for w_val in w_candidates])

        # if we find a real descent direction take the step in its direction
        ind = np.argmin(evals)
        if energySurface(w_candidates[ind]) < energySurface(w):
            # pluck out best descent direction
            d = directions[ind,:]
        
            # take step
            w = w + alpha*d
        
    # record weights and cost evaluation
    weight_history.append(w)
    cost_history.append(energySurface(w))
    return weight_history,cost_history