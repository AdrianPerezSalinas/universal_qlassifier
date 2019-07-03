##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################


#This file provides useful tools checking how good our results are

from circuitery import code_coords, circuit
from fidelity_minimization import fidelity
from weighted_fidelity_minimization import mat_fidelities, w_fidelities
import numpy as np

def _claim(theta, alpha, weight, x, reprs, entanglement, chi):
    """
    This function takes the parameters of a solved problem and one data computes classification of this point
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -y_: the class of x, according to the classifier
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
        
    if chi == 'fidelity_chi':
        y_ = _claim_fidelity(theta, alpha, x, reprs, entanglement)
        
    if chi == 'weighted_fidelity_chi':
        y_ = _claim_weighted_fidelity(theta, alpha, weight, x, reprs, entanglement)
        
    return y_   
        
    
def _claim_fidelity(theta, alpha, x, reprs, entanglement):
    """
    This function is inside _claim for fidelity_chi
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        the class of x, according to the classifier
    """
    theta_aux = code_coords(theta, alpha, x)
    C = circuit(theta_aux, entanglement)
    Fidelities = [fidelity(r, C.psi) for r in reprs]
    
    return np.argmax(Fidelities)


def _claim_weighted_fidelity(theta, alpha, weight, x, reprs, entanglement):
    """
    This function is inside _claim for weighted_fidelity_chi
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        the class of x, according to the classifier
    """
    theta_aux = code_coords(theta, alpha, x)
    fids = mat_fidelities(theta_aux, weight, reprs, entanglement)
    w_fid = w_fidelities(fids, weight)
    return np.argmax(w_fid)
    
def tester(theta, alpha, test_data, reprs, entanglement, chi, weights=None):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -success normalized
    """
    acc = 0
    for i, d in enumerate(test_data):
        x, y = d
        y_ = _claim(theta, alpha, weights, x, reprs, entanglement, chi)
        if y == y_:
            acc += 1
    
    return acc / len(test_data)
        
        
def Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi, weights=None):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -solutions of the classification
        -success normalized
    """
    dim = len(test_data[0][0])
    solutions = np.zeros((len(test_data), dim + 3)) #data  #Esto se podrá mejorar en el futuro
    for i, d in enumerate(test_data):
        x, y = d
        y_ = _claim(theta, alpha, weights, x, reprs, entanglement, chi)
        solutions[i,:dim] = x
        solutions[i, -3] = y
        solutions[i, -2] = y_
        solutions[i, -1] = int(y == y_)
        
    acc = np.sum(solutions[:, -1]) / (i + 1)
    
    return solutions, acc

