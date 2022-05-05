#Strucutre prediction software 
#THIS MAKES FIGURE 7 in GOSH paper

#Import needed modules
import numpy as np
from scipy.optimize import basinhopping
import optimizeLib
from optimizeLib import coordinate_search, random_search, discretised, energySurface
from structurePredictDiscrete import coordinate_searchQ, isGroundEnergyD, randomConfigQ

class atom:
    def __init__(self, typeA,ind):
        self.indx = ind
        self.typeAtom = typeA
        if typeA == 'A':
            self.eps = 1
            self.sig = 1
        else:
            #Atom of type B
            self.eps = 1
            self.sig = 1
            
        #For some tests I had sigma at 1 and 5 and then eps at 1 and 3

def randomConfig (nAtoms, xbound):
    #Input the number of atoms and the x bound for the function
    #Output: a numpy array of n random x coordinates AND the atomlist
    #Atomlist is a length of n atom class
    
    #I have it so the x bound is from -xbound to xbound and so on
    positionList = []
    atomList = []
    typeAtom = True #Goes A,B,A,...
    for i in range(nAtoms):
        #Make a random position
        positionList.append(np.random.uniform(-xbound, xbound))
        #Add the atom
        if typeAtom:
            atomList.append( atom('A', i) ) #Append Atom A
        else:
            atomList.append( atom('B', i) ) #Append Atom B
        typeAtom = not(typeAtom)
        
    return np.array(positionList), atomList
    #Change to numpy just for testing

def isGroundConfig(atomList):
    #Input: A position list of the atoms coordinates.
    #Output: boolean corresponding to if the list of atoms are in an optimal binary configuration
    typeAtom = True
    #A is true, B is false
    newAtomList = []
    for i in atomList:
        newAtomList.append( (i, typeAtom) )
        typeAtom = not(typeAtom)
        
    sorted_by_first = sorted(newAtomList, key=lambda tup: tup[0])
    #print("by first: ", sorted_by_first)
    isBinaryV = True
    for i in range((len(sorted_by_first)-1)):
        if sorted_by_first[i][1] == sorted_by_first[i+1][1]:
            isBinaryV = False
            
    return isBinaryV

def isGroundEnergy(inList):
    #Input: A position list of the atoms coordinates.
    #Output: boolean corresponding to if the list of atoms have the optimal energy
    
    #Generate strucutre with the ideal energy.
    idealPositionList = []
    atomList = []
    typeAtom = True #Goes A,B,A,...
    for i in range(len(inList)):
        #Make a random position
        idealPositionList.append( pow(2, 1/6)*optimizeLib.sigma*i )
        #Add the atom
        if typeAtom:
            atomList.append( atom('A', i) ) #Append Atom A
        else:
            atomList.append( atom('B', i) ) #Append Atom B
        typeAtom = not(typeAtom)  
        
    idealEnergy = energySurface(idealPositionList)
    
    #Then compare this with the actually energy
    delta = 0.48 * len(inList)
    trueEnergy = energySurface(inList)
    if abs(idealEnergy - trueEnergy) < delta:
        return True
    else:
        return False

#Give the scores for 
basinScore, randomScore, discreteScore, discreteScore2, discreteScore3 = 0,0,0,0,0

globalPos, atomList = randomConfig(6, 100)
globalPos= np.array([1,2,3,4,5,6])
#print( isGroundConfig(globalPos) )
#print( isGroundEnergy(globalPos) )

print("Running loop")

#Global variable for the max number of iterations
max_its=100
for i in range(1,7):
    #Generate random configuration
    numAtoms = 2*i + 1
    
    basinScore, randomScore, discreteScore, discreteScore2 = 0,0,0,0
    quantizedScore = 0
    
    for j in range(100):
        #Generate random configuration to use
        globalPos, atomList = randomConfig(numAtoms, 4*numAtoms)
        '''
        #Perform basin hopping minimization
        minEn = basinhopping(energySurface, globalPos) #Returns a OptimizeResult object
        '''
        
        #Random search global optimiser. Encoding 1 not rounded
        #Set parameters
        alpha_choice = 0.5; w = globalPos; num_samples = 500; #max_its = 100;
        #Run random search
        weight_history,cost_history = random_search(alpha_choice,max_its,w,num_samples)
        '''
        #Random discrete search global optimiser. Encoding 1
        #Set parameters
        alpha_choice = 1; w = globalPos;
        #Run coordinate search algorithm 
        weight_history_3,cost_history_3 = coordinate_search(alpha_choice,max_its,w)
        
        alpha_choice = 0.1; w = globalPos;
        #Run coordinate search algorithm 
        weight_history_4,cost_history_4 = coordinate_search(alpha_choice,max_its,w)
        '''
        #Run quantized. Encoding 2 algorithm
        output = randomConfigQ(numAtoms, 4*numAtoms)
        weight_historyQ,cost_historyQ = coordinate_searchQ(output[0], output[1], max_its)  
        '''
        if isGroundEnergy(minEn.x):
            basinScore += 1
        '''
        if isGroundEnergy(weight_history[max_its]):
            randomScore += 1
        '''
        if isGroundEnergy(weight_history_3[max_its]):
            discreteScore += 1
        
        if isGroundEnergy(weight_history_4[max_its]):
            discreteScore2 += 1
        '''
        if isGroundEnergyD(numAtoms, weight_historyQ[max_its][0], weight_historyQ[max_its][1]):
            quantizedScore += 1
            
        
    print("Number of atoms: ", 2*i + 1)
    print(basinScore)
    print("A full random search of encoding 1: ", randomScore)
    print("Encoding 1 with rounding of 1:", discreteScore)
    print(discreteScore2)
    print("Encoding 2 one I quantize in the end:", quantizedScore)
'''
'''

#COME UP WITH BETTER NAMES AND CLARIFY
#ADD TO NOTEBOOK
#MSG CHRIS
#LOOK INTO USING GITHUB
  
print("Done!")