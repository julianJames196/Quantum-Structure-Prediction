#Structure prediction software 

#Import needed modules
import numpy as np
from scipy.optimize import basinhopping
import optimizeLib
from optimizeLib import LJ, LJMod, coordinate_search, random_search, energySurface

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

def LJAtom(atom1, atom2, dis):
    #In put 2 atoms and the distance between them
    #Return the energy given by the lennard jones potential
    #For AA and BB atoms uses the modified potential
    #For AB used the regular lennard jones. Bascially like an ionic crystal
    
    if atom1.typeAtom == atom2.typeAtom:
        #The same atom so interactions are repulsive
        return LJMod( atom1.eps , atom1.sig , dis)
    else:
        #Atoms are different so use a typical lennard jones potential
        return LJ( 0.5*(atom1.eps + atom2.eps) , 0.5*(atom1.sig + atom2.sig) , dis)

def randomConfig (nAtoms, xbound):
    #Input the number of atoms and the x bound for the function
    #Output: a numpy array of n random x coordinates AND the atomlist
    
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
    
    #ERROR IN IDEAL POSITION LIST
    print(idealPositionList)    
        
    idealEnergy = energySurface(idealPositionList)
    
    #Then compare this with the actually energy
    delta = 0.5 * len(inList)
    trueEnergy = energySurface(inList)
    if abs(idealEnergy - trueEnergy) < delta:
        return True
    else:
        return False

#Generate random configuration
numAtoms = 11

globalPos, atomList = randomConfig(numAtoms, 3*numAtoms)

'''
#Manually set
globalPos = np.array([24.47368029031403, 17.0385, 24.923796616911872, -9.6098, -23.4184, -21.2254, -15.92781115, -6.331])
'''

#Output intial random configuration
for i in range(numAtoms):
    print("Atom", atomList[i].typeAtom, "#", atomList[i].indx, " x pos: ", globalPos[i])
    
print("Inital energy:", energySurface(globalPos) )

'''
#Perform Basinhopping minimization
minEn = basinhopping(energySurface, globalPos) #Returns a OptimizeResult object

print("Final position:", minEn.x)
print("Final min energy:", minEn.fun)
print("Number of evaluations of the objective functions:", minEn.nfev)
print("Number of iterations performed by the optimizer:", minEn.nit)

#Shift the positons
finalPos = np.add(minEn.x , np.full((1, numAtoms), -min(minEn.x)- (1.12 * numAtoms*0.5) ))
print("Shifted Final positions:", finalPos[0] )


#Random search global optimiser
print("")
print("Running random search global optimiser")

#Set parameters
alpha_choice = 1; w = globalPos; num_samples = 500; max_its = 100;
#Run random search
weight_history,cost_history = random_search(alpha_choice,max_its,w,num_samples)

#Print the final energy at the end
print(cost_history[max_its])
#Print the final posiitons
print(weight_history[max_its])
'''

#Random discrete search global optimiser
print("")
print("Running discrete random search global optimiser")

#Set parameters
alpha_choice = 0.1; w = globalPos; max_its = 20;
#Run coordinate search algorithm
weight_history_2,cost_history_2 = coordinate_search(alpha_choice,max_its,w)

print(cost_history_2[0])
print("Intial:", weight_history_2[0])
#print("Min 1:", weight_history_2[1])
#print("Min 2:", weight_history_2[2])

#Print the final energy at the end
print(cost_history_2[max_its])
#Print the final posiitons
print(weight_history_2[max_its])

print("")
'''
print("Is basinhopping binary: ", isBinary(minEn.x))
print("Is global search binary: ", isBinary(weight_history[max_its]))
'''
print("Before discrete minization: ", isBinary(weight_history_2[0]))
print("After discrete minization: ", isBinary(weight_history_2[max_its]))

for i in range(900):
    globalPos, atomList = randomConfig(numAtoms, 3*numAtoms)
    #Run coordinate search algorithm
    weight_history_2,cost_history_2 = coordinate_search(alpha_choice,max_its,w)
    if isBinary(weight_history_2[0]) != isBinary(weight_history_2[max_its]):
        #Print result of when beating bound
        print("Beat")
        print(weight_history_2[0])
        print(cost_history_2[0])
        print(weight_history_2[max_its])
        print(cost_history_2[max_its])

#TEST INCONCLUSIVE TRUE TO DEFINYION OF GROUND STATE
print("Done")