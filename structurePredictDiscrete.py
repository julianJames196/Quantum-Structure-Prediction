#Strucutre prediction software 
#This uses similar ideas to second quantization to do discrete optimization
#Represent the atom positions with binary numbers. Will make easy to quantize.

#Import needed modules
import numpy as np
import math
from itertools import combinations, product
import optimizeLib
from optimizeLib import LJ, LJMod, energySurface

#Global system variables
#A global variable for the distance between adjacent bins. Sets the physical scale
binSize = 0.5*pow(2, 1/6)*optimizeLib.sigma #Ideal distance is 2 bins away

def randomConfigQ(nAtoms, lengthL):
    #Input the number of atoms and the number of bins
    #Output: 2 binary arrays of our A and B atoms
    
    numA = math.ceil(nAtoms / 2)
    numB = math.floor(nAtoms / 2)
    aList = []
    bList = []
    #Intialize them to be of length lengthL all be empty
    for i in range(lengthL):
        aList.append(False)
        bList.append(False)
        
    #Create the list of A atoms
    for i in range(numA):
        #aList[math.floor( np.random.random() * lengthL)] = True
        index = math.floor( np.random.random() * lengthL)
        while aList[index]:
            index = math.floor( np.random.random() * lengthL)
        aList[index] = True
        
    #Create the list of B atoms 
    for i in range(numB):
        index = math.floor( np.random.random() * lengthL)
        while bList[index]:
            index = math.floor( np.random.random() * lengthL)
        bList[index] = True
        
    return aList, bList
    #Collisions are fixed with the while loop
    
def energySurfaceQ(atomA, atomB):
    #Input: The 2 atom lists#
    #Output: the potential energy of this configuration
    
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
            energy += LJ(Dis) #REPULSION POTENTIAL
    
    return energy

def isGroundEnergyD(nAtoms, atomA, atomB):
    #Input: atomA positions and atomB positions
    #Output: Check if the configuration is binary/optimal
    
    #Generate continous structure with the ideal energy.
    idealPositionList = []
    for i in range(nAtoms):
        #Make a random position
        idealPositionList.append( pow(2, 1/6)*optimizeLib.sigma*i )
        
    idealEnergy = energySurface(idealPositionList)
    
    #Check this energy with the actually energy of the input state. 
    delta = 0.8 * nAtoms
    trueEnergy = energySurfaceQ(atomA, atomB)
    if abs(idealEnergy - trueEnergy) < delta:
        return True
    else:
        return False
    
def whereAtoms(qAtomList):
    #Input: a binary array with the position of the items
    #Output a tuple with the first tuple saying where the atoms and second inverse
    isAtom = []
    notAtom = []
    for i in range(len(qAtomList)):
        if qAtomList[i]:
            isAtom.append(i)
        else:
            notAtom.append(i)
            
    return isAtom, notAtom

'''
def coordinate_searchQ():
    #Step one take the number and convert to an array.
    #2, check the energy of the configuration
    #3, do bit flips to minimize
    #4, end when in local minimum
'''

#Uses encoding 2 which is what I use with my VQE. 
#Maybe make a python that inputs 2 numbers and removes at one and creates at another.
#creation and annihilation operator essentially. 

#Zero order bit flip search
def coordinate_searchQ(atomA, atomB, max_its):
    # run coordinate search
    weight_history = []         # container for weight history
    cost_history = []           # container for corresponding cost function history
    #Maybe have an alpha
    typeAtom = False #To decide is flipping A or B. 
    for k in range(1,max_its+1):
        #Record weights and cost evaluation
        weight_history.append((atomA, atomB))
        cost_history.append(energySurfaceQ(atomA, atomB))
        typeAtom = not(typeAtom)
        #See if in this loop we are changing A or B depending on typeAtom variable ADD
        if typeAtom:    
            #Find where all the atoms in the list we have choosen.
            isAtom, notAtom = whereAtoms(atomA) #Tenary operator to decide which list
            #So we can then find all bit flips
            evals = [] #List all the energy possibilities
            evalFlips = [] #A tuple of all the chnages made
            for i in isAtom:
                #Loop through all atom combos
                atomA[i] = False #Annihilate atomA
                for j in notAtom:
                    atomA[j] = True #Create the atom
                    evals.append(energySurfaceQ(atomA, atomB)) #Evaulate the energy of flip
                    evalFlips.append( (i,j) )
                    atomA[j] = False #Unbitflip
                atomA[i] = True
            
            # Find best bit flip and do if we decrease the energy
            ind = np.argmin(evals)
            if evals[ind] < energySurfaceQ(atomA, atomB):
                # pluck out best bit flip
                flip = evalFlips[ind]
            
                # Do the bit flip
                atomA[flip[0]] = False
                atomA[flip[1]] = True
        else:
            #Going through and flipping atom Bs.
            #Find where all the atoms in the list we have choosen.
            isAtom, notAtom = whereAtoms(atomB) #Tenary operator to decide which list
            #So we can then find all bit flips
            evals = [] #List all the energy possibilities
            evalFlips = [] #A tuple of all the chnages made
            for i in isAtom:
                #Loop through all atom combos
                atomB[i] = False #Annihilate atomB
                for j in notAtom:
                    atomB[j] = True #Create the atom
                    evals.append(energySurfaceQ(atomA, atomB)) #Evaulate the energy of flip
                    evalFlips.append( (i,j) )
                    atomB[j] = False #Unbitflip
                atomB[i] = True
            
            # Find best bit flip and do if we decrease the energy
            ind = np.argmin(evals)
            if evals[ind] < energySurfaceQ(atomA, atomB):
                # pluck out best bit flip
                flip = evalFlips[ind]
            
                # Do the bit flip
                atomB[flip[0]] = False
                atomB[flip[1]] = True
        
    # record final weights and cost evaluation
    weight_history.append((atomA, atomB))
    cost_history.append(energySurfaceQ(atomA, atomB))
    
    return weight_history,cost_history
'''
#- atoms, -- bins
output = randomConfigQ(3, 20)
#output = [[True,False,False,False,False],[False,False,False,False,True]] #Manual test
print(output[0])
print(output[1])
print(energySurfaceQ(output[0], output[1]))
print("Running optimization")

max_its = 6
weight_history,cost_history = coordinate_searchQ(output[0], output[1], max_its)
print("Outputs")
print(weight_history[max_its][0])
print(weight_history[max_its][1])
print(cost_history[max_its])
    '''
print("Done")