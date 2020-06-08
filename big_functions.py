##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

#This file is a file taking many different functions from other files and mixing them all together
# so that the usage is automatized

from data_gen import data_generator
from problem_gen import problem_generator, representatives
from fidelity_minimization import fidelity_minimization
from weighted_fidelity_minimization import weighted_fidelity_minimization
from test_data import Accuracy_test, tester
from save_data import write_summary, read_summary, name_folder, samples_paint, samples_paint_worldmap, laea_x, laea_y
from save_data import write_epochs_file, write_epoch, close_epochs_file, create_folder, write_epochs_error_rate
import numpy as np
import matplotlib.pyplot as plt
from circuitery import code_coords, circuit
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

def minimizer(chi, problem, qubits, entanglement, layers, method, name,
              seed = 30, epochs=3000, batch_size=20,  eta=0.1):
    """
    This function creates data and minimizes whichever problem (from the selected ones) 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    np.random.seed(seed)
    data, drawing = data_generator(problem)
    if problem == 'sphere':
        train_data = data[:500] 
        test_data = data[500:]
    elif problem == 'hypersphere':
        train_data = data[:1000] 
        test_data = data[1000:]
    else:
        train_data = data[:200]
        test_data = data[200:]
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
        theta, alpha, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, f = fidelity_minimization(theta, alpha, train_data, reprs,
                                            entanglement, method, 
                                            batch_size, eta, epochs)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, 0, f, acc_train, acc_test, seed, epochs=epochs)
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        theta, alpha, weight, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, weight, f = weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs,
                                            entanglement, method)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi, weights=weight)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi, weights=weight)
        write_summary(chi, problem, qubits, entanglement, layers, method, name,
              theta, alpha, weight, f, acc_train, acc_test, seed, epochs=epochs)

    
def painter(chi, problem, qubits, entanglement, layers, method, name, 
            seed = 30, standard_test = True, samples = 4000, bw = False, err = False):
    """
    This function takes written text files and paint the results of the problem 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -standard_test: Whether we want to paint the set test used for checking when minimizing. If True, seed and samples are not taken in account
        -samples: number of samples of the test set
        -bw: painting in black and white
    OUTPUT:
        This function has got no outputs, but a file containing the representation of the test set is created
    """
    np.random.seed(seed)
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        
    if standard_test == True:
        data, drawing = data_generator(problem)
        if problem == 'sphere':
            test_data = data[500:]
        elif problem == 'hypersphere':
            test_data = data[1000:]
        else:
            test_data = data[200:]
            
    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples = samples)
            
    if problem in ['circle','wavy circle','sphere', 'non convex', 'crown', 'hypersphere']:
        classes = 2
    if problem in ['tricrown']:
        classes = 3
    elif problem in ['3 circles','wavy lines','squares']:
        classes = 4
        
    reprs = representatives(classes, qubits_lab)
    
    params = read_summary(chi, problem, qubits, entanglement, layers, method, name)
    
    if chi == 'fidelity_chi':
        theta, alpha = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi)
        
    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs,
                                           entanglement, chi, weights = weight)

    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    samples_paint(problem, drawing, sol_test, foldname, name, bw)

def paint_world(chi, problem, qubits, entanglement, layers, method, name,
            seed = 30, standard_test = True, samples = 4000, bw = False, err = False):
    np.random.seed(seed)

    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1

    if standard_test == True:
        data, drawing = data_generator(problem)
        if problem == 'sphere':
            test_data = data[500:]
        elif problem == 'hypersphere':
            test_data = data[1000:]
        else:
            test_data = data[200:]

    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples=samples)

    if problem in ['circle', 'wavy circle', 'sphere', 'non convex', 'crown', 'hypersphere']:
        classes = 2
    if problem in ['tricrown']:
        classes = 3
    elif problem in ['3 circles', 'wavy lines', 'squares']:
        classes = 4

    reprs = representatives(classes, qubits_lab)

    params = read_summary(chi, problem, qubits, entanglement, layers, method, name)

    if chi == 'fidelity_chi':
        theta, alpha = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi)

    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs,
                                           entanglement, chi, weights=weight)

    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    angles = np.zeros((len(sol_test), 2))
    for i, x in enumerate(sol_test[:, :2]):
        theta_aux = code_coords(theta, alpha, x)
        C = circuit(theta_aux, entanglement)
        angles[i, 0] = np.arccos(np.abs(C.psi[0])**2 - np.abs(C.psi[1])**2) - np.pi/2
        angles[i, 1] = np.angle(C.psi[1] / C.psi[0])
        print(angles[i])

    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5, vmax=np.max(sol_test[:, -3]) + .5)

        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1, vmax=1.1)

    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.1, vmax=np.max(sol[:, -3]) + .1)

        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1, vmax=1.1)

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(laea_x(np.pi, np.linspace(0, np.pi)), laea_y(np.pi, np.linspace(0, np.pi)), color='k')
    ax[0].plot(laea_x(-np.pi, np.linspace(0, -np.pi)), laea_y(-np.pi, np.linspace(0, -np.pi)), color='k')
    ax[1].plot(laea_x(np.pi, np.linspace(0, np.pi)), laea_y(np.pi, np.linspace(0, np.pi)), color='k')
    ax[1].plot(laea_x(-np.pi, np.linspace(0, -np.pi)), laea_y(-np.pi, np.linspace(0, -np.pi)), color='k')
    ax[0].scatter(laea_x(angles[:, 1], angles[:, 0]), laea_y(angles[:, 1], angles[:, 0]), c=sol_test[:, -2],
                  cmap=colors_classes, s=2, norm=norm_class)
    ax[1].scatter(laea_x(angles[:, 1], angles[:, 0]), laea_y(angles[:, 1], angles[:, 0]), c=sol_test[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)
    plt.show()


def SGD_step_by_step_minimization(problem, qubits, entanglement, layers, name, 
                                  seed = 30, epochs = 3000, batch_size = 20, eta = .1, err=False):
    """
    This function creates data and minimizes whichever problem using a step by step SGD and saving all results from accuracies for training and test sets
    INPUT:
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -error_rates: accuracies for training and test sets as flat arrays
    """
    chi = 'fidelity_chi'
    method = 'SGD'
    
    np.random.seed(seed)
    data, drawing = data_generator(problem, err=err)
    if problem == 'sphere':
        train_data = data[:500] 
        test_data = data[500:]
    elif problem == 'hypersphere':
        train_data = data[:1000] 
        test_data = data[1000:]
    else:
        train_data = data[:200]
        test_data = data[200:]
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
    
    theta, alpha, reprs = problem_generator(problem, qubits, layers, chi,
                                            qubits_lab=qubits_lab)
    accs_test=[]
    accs_train=[]
    chis=[]
    acc_test_sol = 0
    acc_train_sol = 0
    fid_sol = 0
    best_epoch = 0
    theta_sol = theta.copy()
    alpha_sol = alpha.copy()
    
    file_text = write_epochs_file(chi, problem, qubits, entanglement, layers, method, name)
    for e in range(epochs):
        theta, alpha, fid = fidelity_minimization(theta, alpha, train_data, reprs,
                                            entanglement, method, batch_size, eta, 1)
        
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi)
        accs_test.append(acc_test)
        accs_train.append(acc_train)
        chis.append(fid)
        
        write_epoch(file_text, e, theta, alpha, fid, acc_train, acc_test)
    
        if acc_test > acc_test_sol:
            
            acc_test_sol = acc_test
            acc_train_sol = acc_train
            fid_sol = fid
            theta_sol = theta
            alpha_sol = alpha
            best_epoch = e

    close_epochs_file(file_text, best_epoch)
    write_summary(chi, problem, qubits, entanglement, layers, method, name,
          theta_sol, alpha_sol, None, fid_sol, acc_train_sol, acc_test_sol, seed, epochs)
    write_epochs_error_rate(chi, problem, qubits, entanglement, layers, method, name, 
                      accs_train, accs_test)
    
def overlearning_paint(chi, problem, qubits, entanglement, layers, method, name):
    """
    This function takes overlearning functions and paints them
    INPUT:
        -chi: cost function, just 'fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, 'SGD'
        -name: a name we want for our our files to be save with
    OUTPUT:
        This function has got no outputs, but saves a picture with the information of the overlearning rates
    """
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    create_folder(foldname)
    filename_train = foldname + '/' + name + '_train.txt'
    filename_test = foldname + '/' + name + '_test.txt'
    
    train_err_rate = np.loadtxt(filename_train)
    test_err_rate = np.loadtxt(filename_test)
    fig, ax = plt.subplots()
    ax.plot(range(len(train_err_rate)), train_err_rate, label = 'Training set')
    ax.plot(range(len(test_err_rate)), test_err_rate, label = 'Test set')
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_ylabel('Error rate', fontsize=16)
    ax.legend()
    filename = foldname + '/' + name + '_overlearning'
    fig.savefig(filename)
    plt.close('all')
    
   
