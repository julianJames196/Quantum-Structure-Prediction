#!/usr/bin/env python
# coding: utf-8

# In[8]:


#The quantum algorithm
import numpy as np
from matplotlib import pyplot as plt
import sys #This allows us to exit loops

from qiskit import *
from qiskit.tools.visualization import plot_histogram

from qiskit.circuit.library import RealAmplitudes
#RealAmplitudes is an Ansatz for the VQE

from scipy.optimize import minimize
from itertools import combinations, product
#My library
import optimizeLib
from optimizeLib import LJ, LJMod, energySurface

#Global system variables
binSize = 0.5*pow(2, 1/6)*optimizeLib.sigma #A global variable for the distance between adjacent bins. 
#Sets the physical scale

n = 14 #Number of qubits in the circuit. The number of bins. 
#n will likely have to be even
maxiter = 100
nshots = 1e3 #Is the number to be varied

#A method to take a binary string and calculate energy
def energy_int(integerIn):
    #Input: an integer
    #Out the energy of the string using encoding 2
    #Builds on energySurfaceQ in strucutrePredictQuantized
    
    #Take the input and convert to a binary array
    binArray = [bool(int(x)) for x in str( "{0:b}".format(integerIn)  )]
    #Need to make it of length n
    temp = np.full( n-len(binArray), False )
    binArray = np.concatenate((temp, binArray))
    #First split the array in half
    l = int(n /2)
    atomA = binArray[:l] #First half of the array
    atomB = binArray[l:]
    
    #Run the energySurfaceQ function
    #Note len(atomA)=len(atomB)=Number of bins pre specified
    energy = 0
    #Go through the atomA list to see repulsion
    for x in combinations(range(len(atomA)), 2):
        #x is a tuple of the form (0,3)
        if atomA[x[0]] and atomA[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJMod(Dis) #REPULSION POTENTIAL
            
        if atomB[x[0]] and atomB[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJMod(Dis) #REPULSION POTENTIAL
    
    #Go through both to find attraction. 
    for x in product(range(len(atomB)), repeat = 2):
        if atomA[x[0]] and atomB[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJ(Dis) #Regular Lennard-Jones
    
    #Output energy
    return energy
    
def energy_from_full_dist(pos):
    #Input: pos which is a dictionary of length 2^n
    ensum= 0
    for j in pos:
        #j are all the keys and are 0,1,2,...,(2^n -1)
        #ensum += energy_int(j)
        ensum += (pos[j]*energy_int(j)*0.1) #Need to weight by different values

    return ensum

#######################################
def build_rycnot(n, depth, params):
    """RY CNOT ansaztz"""
    if len(params) != n*depth:
        print("ERROR!, Too few parameters")
        sys.exit()

    ansatz = RealAmplitudes(n, entanglement='linear', reps=int(depth)-1 , insert_barriers =True)

    if len(params) == ansatz.num_parameters_settable:
        ansatz.ordered_parameters = params
    else:
        print('ERROR!, Number of parameters to vary doesnt equal the number given' )
        sys.exit()

    return ansatz

#from shots counts to distribution
def find_distribution(n, counts, NSHOTS):
    #yes this is exponentially costly written like that..
    N = 1<<n
    distr = {}
    for n in range(N):
        try:
            distr[n] = counts[n]
        except:
            distr[n] = 0
    return distr

def qasm_distribution(n, par, NSHOTS = 1e4):
    """execute the circuit and return the dictionary with the counts"""
    """ NSHOTS is the number to be varied """
    N = 1<<n
   
    qc = QuantumCircuit(n)
    
    depth = int(len(par)/n)
    
    ansatz = build_rycnot(n, depth, par)
    qc.compose(ansatz, inplace = True)
    qc.measure_all()

    #backend definition and run
    backend = Aer.get_backend('qasm_simulator')
    qcc = transpile(qc, backend)
    counts = backend.run(qcc, shots = NSHOTS).result().get_counts()
    
    #coversion from counts "000010" label into integer j label (binary)
    int_counts = counts.int_outcomes()
    
    #create normalized histogram
    #pos has length 2^n
    pos = find_distribution(n, int_counts, NSHOTS)

    return pos

#Work on this
def cost_function(params):            
    '''from circuit parameters to cost function'''
              
    '''include something like'''
    
    pos = qasm_distribution(n, params)
    cost_function = energy_from_full_dist( pos)         
                  
    return cost_function

def isGroundEnergyQ(nAtoms, binaryString):
    #Input: number of atoms, the inputted binary string
    #Output: Check if the configuration is binary/optimal
    
    #Generate continous structure with the ideal energy.
    idealPositionList = []
    for i in range(nAtoms):
        #Make a random position
        idealPositionList.append( pow(2, 1/6)*optimizeLib.sigma*i )
        #List goes A,B,A,B,...
        
    idealEnergy = energySurface(idealPositionList)
    
    #Check this energy with the actually energy of the input state. 
    delta = 0.15 * nAtoms
    trueEnergy = energy_int(binaryString)
    if abs(idealEnergy - trueEnergy) < delta:
        return True
    else:
        return False
    
def numAnumB(binaryString):
    #Input a string which we convert to binary and using our encoding figure out how many a and b atoms.
    #Output: 2 integers for the number of A atoms and B atoms. 
    
    #Take the input and convert to a binary array
    binArray = [bool(int(x)) for x in str( "{0:b}".format(binaryString)  )]
    #Need to make it of length n
    temp = np.full( n-len(binArray), False )
    binArray = np.concatenate((temp, binArray))
    #First split the array in half
    l = int(n /2)
    atomA = binArray[:l] #First half of the array
    atomB = binArray[l:]
    
    atomANum = 0
    atomBNum = 0
            
    for i in range(len(atomA)):
        if atomA[i]:
            atomANum += 1
        
        if atomB[i]:
            atomBNum += 1
            
    return atomANum, atomBNum

#Takes the highest confidence values and prints out the keys.
def dictionary_confidence(d,conf):
    #Input a dictionary and a confidence value
    #Output: an array of key from the dictionary fitting the confidence criterion
    sumVal = sum(d.values())
    #Sort the dictionary by values. Need items to keep both values
    ar = sorted(d.items(), key=lambda x: x[1], reverse=True)
    sumV = 0
    keyArray = []
    index = 0
    while sumV < (sumVal*conf) and index < (len(d)-1):
        sumV += ar[index][1] #Add the value in the dictionary. Value is second
        keyArray.append(ar[index][0]) #Apped the dicitonary key
        index += 1 #Increase the index
        
    return keyArray

#Main for this part
'''
param = np.random.rand(4*n) #n given above as global var
pos = qasm_distribution(n, param) #A 2^n length dictionary

print(pos)
print("Matplotlib")
#plt.hist(pos)
plt.bar(pos.keys(), pos.values(), 1.0, color='g')
plt.title("Inital values before optimization step")
plt.show()


print("Round 2")
pos = qasm_distribution(n, result.x)

print(pos)
print("Plotting")
#plot_histogram(pos, bar_labels = False)

plt.bar(pos.keys(), pos.values(), 1.0, color='g')
plt.show()

#Include additonal sub routine parameters in args
print("Calculting result")
result = minimize(cost_function, param,
                            method='COBYLA',tol=1e-6, options={'maxiter':maxiter} )
'''
param = np.random.rand(4*n) #n given above as global var
for i in range(4):
    print(i)
    #Running the VQE step x number of times
    pos = qasm_distribution(n, param) #A 2^n length dictionary
    
    #Plot the pos
    plt.bar(pos.keys(), pos.values(), 1.0, color='g')
    title = str("After "+ str(i+1) + " run through the VQE anstaz")
    plt.title(title)
    plt.ylabel("Counts")
    plt.xlabel("Binary string")
    plt.show()
    
    #Run classical optimzer
    result = minimize(cost_function, param,
                            method='COBYLA',tol=1e-3, options={'maxiter':maxiter} )
    #Update parameters
    param = result.x

#Run the values a final time to get a graph
pos = qasm_distribution(n, param)

plt.bar(pos.keys(), pos.values(), 1.0, color='g')
plt.title("Final")
plt.show()
#print("Final positions", pos)

#Find if the solutions found are optimal
array = dictionary_confidence(pos,100) #These are the XX% confidence values. Length <= 2^n

#Loop through all the XX% confidence values.
#Check it fits the criteria for having Number of A atoms = Number of B atoms -1 (vica versa) Maybe A=B works
#Find if it's the optimal solution for 0,1,2,... atoms
#Increase the increament count if needed(How one wants to deal with degeneracies). 

twoAtoms = []
threeAtoms = []
fourAtoms = []
indexes = [0,0,0] #Track how many along we are for each atom count
for x in array:
    A,B = numAnumB(x)
    numberAtoms = A + B
    if numberAtoms > 1.1:
        #Only consider none trival optimizations with more than 2 atoms
        if A == B-1 or B == A-1 or A==B:
            #Only consider the ones with ABA or ABBA not AAAB as can never gurantee optimality
            if numberAtoms == 2:
                indexes[0] += 1
                if isGroundEnergyQ(2,x):
                    twoAtoms.append(indexes[0])

            if numberAtoms == 3:
                indexes[1] += 1
                if isGroundEnergyQ(3,x):
                    threeAtoms.append(indexes[1])

            if numberAtoms == 4:
                indexes[2] += 1
                if isGroundEnergyQ(4,x):
                    fourAtoms.append(indexes[2])
        
print("What positions did we get the 2,3,4,... atom optimzation")
print(twoAtoms)
print(threeAtoms)
print(fourAtoms)
print("Done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




