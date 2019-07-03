##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################


#This file provides the minimization for the cheap chi square
from circuitery import code_coords, circuit
import numpy as np
from scipy.optimize import minimize

def weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs, 
                       entanglement, method):
    """
    This function takes the parameters of a problem and computes the optimal parameters for it, using different functions. It uses the weighted fidelity minimization
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: minimization method, to choose between valid methods for function scipy.optimize.minimize]
    OUTPUT:
        -theta: optimized point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: optimized point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -chi: value of the minimization function
    """
    
    params, hypars = _translate_to_scipy(theta, alpha, weight)
    results = minimize(_scipy_minimizing, params, 
                       args = (hypars, train_data, reprs, entanglement),
                       method=method)
    theta, alpha, weight = _translate_from_scipy(results['x'], hypars)
            
    return theta, alpha, weight, results['fun']

def _braket(qState1, qState2):
    """
    This function returns the relativy fidelity of two pure states
    INPUT:
        -2 pure states of the same dimension
    OUTPUT:
        -relative fidelity
    """
    return np.dot(np.conj(qState1), qState2)

def mat_fidelities(theta_aux, weight, reprs, entanglement,
                    return_circuit = False):
    """
    This function takes computes fidelities for a given circuit and weigths
    INPUT: 
        -theta_aux: set of parameters needed for the circuit, alpha is encoded here too. It is an array with shape (qubits, layers, 3)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -return_circuit: boolean varaible, True if the circuit is returned
    OUTPUT:
        -Fidelities: relative fidelities for different labels and qubits
        -C: quantum circuit
    """
    labels = weight.shape[0]
    qubits = weight.shape[1]
    Fidelities = np.empty(weight.shape)
    C = circuit(theta_aux, entanglement)
    for q in range(qubits):
        rdm = C.reduced_density_matrix(q)
        for l in range(labels):
            Fidelities[l, q] = np.real(np.conj(reprs[l]) @ rdm @ reprs[l])
            
    if return_circuit == False:
        return Fidelities
    
    if return_circuit == True:
        return Fidelities, C

def w_fidelities(Fidelities, weight):
    """
    This function weights fidelities for a given circuit and weigths
    INPUT: 
        -Fidelities: relative fidelities for different labels and qubits
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
    OUTPUT:
        -w_fid: weighted fidelities for different labels
    """
    w_fid = np.sum(Fidelities * weight, axis=1)
    return w_fid

def _reduced_density_matrix(a, b, qubit):
    """
    This function computes the partial trace of two quantum states with respect to one qubit
    INPUT:
        -a, b: two quantum states
        -qubit: the qubit we do not want to trace out
        
    OUTPUT:
        -rdm: analogy to the reduced density matrix
    """
    num_qubits = int(np.log2(len(a)))
    rdm = np.zeros((2,2),dtype='complex')
    for i in range(2):
        for j in range(i + 1):
            for k in range(2**(num_qubits-1)):
                S = k%(2**qubit) + 2*(k - k%(2**qubit))
                rdm[i,j] += (a[S + i*2**qubit] * np.conj(b[S + j*2**qubit]))
            rdm[j,i] = np.conj(rdm[i,j])

    return rdm


def _chi(theta, alpha, weight, d, reprs, entanglement):
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits) 
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -chi^2 for data
    """
    x, y = d
    theta_aux = code_coords(theta, alpha, x)
    fids = mat_fidelities(theta_aux, weight, reprs, entanglement)
    w_fid = w_fidelities(fids, weight)
    if len(w_fid) == 4:
        Y = 1 / 3 * np.ones(len(w_fid))
        Y[y] = 1
    if len(w_fid) == 3:
        Y = 1 / 4 * np.ones(len(w_fid))
        Y[y] = 1
    if len(w_fid) == 2:
        Y = np.zeros(len(w_fid))
        Y[y] = 1
    return .5 * np.linalg.norm(w_fid - Y) ** 2


def Av_Chi_Square(theta, alpha, weight, data, reprs, entanglement):
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits) 
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -Averaged chi^2 for data
    """
    Av_Chi = 0
    for d in data:
        Av_Chi += _chi(theta, alpha, weight, d, reprs, entanglement)
        
    return Av_Chi / len(data)


def _translate_to_scipy(theta, alpha, weight):
    """
    This function is a intermediate step for translating theta and alpha to a single variable for scipy.optimize.minimize
    """
    qubits = theta.shape[0]
    layers = theta.shape[1]
    dim = alpha.shape[-1]
    classes = weight.shape[0]
    
    return np.concatenate((theta.flatten(), alpha.flatten(),weight.flatten())), (qubits, layers, dim, classes)

def _translate_from_scipy(params, hypars):
    """
    This function is a intermediate step for getting theta and alpha from a single variable for scipy.optimize.minimize
    """
    (qubits, layers, dim, classes) = hypars
    if dim <= 3:
        theta = params[:qubits * layers * 3]. reshape(qubits, layers, 3)
        alpha = params[qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(qubits, layers, dim)
        weight = params[(qubits * layers * 3 + qubits * layers * dim):].reshape(classes, qubits)
    
    if dim == 4:
        theta = params[:qubits * layers * 6]. reshape(qubits, layers, 6)
        alpha = params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(qubits, layers, dim)
        weight = params[(qubits * layers * 6 + qubits * layers * dim):].reshape(classes, qubits)
    
    return theta, alpha, weight

def _scipy_minimizing(params, hypars, train_data, reprs, entanglement):
    """
    This function returns the chi^2 function for using scipy
    INPUT:
        -params: theta and alpha inside the same variable
        -hypars: hyperparameters needed to rebuild theta and alpha
        -train_data: training dataset for the classifier
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        - Av_chi_square, which is the function we want to minimize
    """
    theta, alpha, weight = _translate_from_scipy(params, hypars)
    return Av_Chi_Square(theta, alpha, weight, train_data, reprs, entanglement)
