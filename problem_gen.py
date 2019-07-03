##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file creates the problems and their settings
import numpy as np

def problem_generator(problem, qubits, layers, chi, qubits_lab=1):
    """
    This function generates everything needed for solving the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account

        
    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
        -reprs: variable encoding the label states of the different classes
    """
    chi = chi.lower()
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
        
    if chi == 'weighted_fidelity_chi' and qubits_lab != 1: 
        qubits_lab = 1
        print('WARNING: number of qubits for the label states has been changed to 1')
    
    problem = problem.lower()
    if problem == 'circle':
        theta, alpha, reprs = _circle(qubits, layers, qubits_lab, chi)
    elif problem == '3 circles':
        theta, alpha, reprs = _3_circles(qubits, layers, qubits_lab, chi)
    elif problem == 'wavy lines':
        theta, alpha, reprs = _wavy_lines(qubits, layers, qubits_lab, chi)
    elif problem == 'squares':
        theta, alpha, reprs = _squares(qubits, layers, qubits_lab, chi)
    elif problem == 'sphere':
        theta, alpha, reprs = _sphere(qubits, layers, qubits_lab, chi)
    elif problem == 'non convex':
        theta, alpha, reprs = _non_convex(qubits, layers, qubits_lab, chi)
    elif problem == 'crown':
        theta, alpha, reprs = _crown(qubits, layers, qubits_lab, chi)
    elif problem == 'tricrown':
        theta, alpha, reprs = _tricrown(qubits, layers, qubits_lab, chi)
    elif problem == 'hypersphere':
        theta, alpha, reprs = _hypersphere(qubits, layers, qubits_lab, chi)
        
    else:
        raise ValueError('Problem is not valid')
        
    if chi == 'fidelity_chi':
        return theta, alpha, reprs
    elif chi == 'weighted_fidelity_chi':
        weights = np.ones((len(reprs), qubits))
        return theta, alpha, weights, reprs

#All these are auxiliary functions for problem_generator
def _circle(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs
        
def _3_circles(qubits, layers, qubits_lab, chi):
    classes = 4
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs

def _wavy_lines(qubits, layers, qubits_lab, chi):
    classes = 4
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs        

def _squares(qubits, layers, qubits_lab, chi):
    classes = 4
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs        

def _non_convex(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs        

def _crown(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs        

def _tricrown(qubits, layers, qubits_lab, chi):
    classes = 3
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs      

def _sphere(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 3)
    return theta, alpha, reprs   

def _hypersphere(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 6)
    alpha = np.random.rand(qubits, layers, 4)
    return theta, alpha, reprs   
        
def representatives(classes, qubits_lab):
    """
    This function creates the label states for the classification task
    INPUT: 
        -classes: number of classes of our problem
        -qubits_lab: how many qubits will store the labels
    OUTPUT:
        -reprs: the label states
    """
    reprs = np.zeros((classes, 2**qubits_lab), dtype = 'complex')
    if qubits_lab == 1:
        if classes == 0:
            raise ValueError('Nonsense classifier')
        if classes == 1:
            raise ValueError('Nonsense classifier')
        if classes == 2:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
        if classes == 3:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / 2, np.sqrt(3) / 2])
            reprs[2] = np.array([1 / 2, -np.sqrt(3) / 2])
        if classes == 4:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
            reprs[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
            reprs[3] = np.array([1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
        if classes == 6:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
            reprs[2] = 1 / np.sqrt(2) * np.array([1, 1])
            reprs[3] = 1 / np.sqrt(2) * np.array([1, -1])
            reprs[4] = 1 / np.sqrt(2) * np.array([1, 1j])
            reprs[5] = 1 / np.sqrt(2) * np.array([1, -1j])

    if qubits_lab == 2:
        if classes == 0:
            raise ValueError('Nonsense classifier')
        if classes == 1:
            raise ValueError('Nonsense classifier')
        if classes == 2:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 0, 0, 1])
        if classes == 3:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
        if classes == 4:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
            reprs[3] = np.array([0, 0, 0, 1])
            
    return reprs
