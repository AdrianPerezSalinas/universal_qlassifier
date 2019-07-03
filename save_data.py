##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################


#This file provides useful tools for painting and saving data according to the problem, 
# the minimization style, the number of qubits and layers.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap 
from matplotlib.colors import Normalize

def write_summary(chi, problem, qubits, entanglement, layers, method, name,
          theta, alpha, weights, chi_value, acc_train, acc_test, seed, epochs):
    """
    This function takes some informations of a given problem and saves some text files 
    with this information and the parameters which are solution of the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose between
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. Must be an array with shape (classes, qubits)
        -chi_value: Value of the cost function after minimization
        -acc_train: accuracy for the training set
        -acc_test: accuracy for the test set
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    create_folder(foldname)
    file_text = open(foldname + '/' + name + '_summary.txt','w')
    file_text.write('\nFigur of merit = '+chi)
    file_text.write('\nProblem = ' + problem)
    file_text.write('\nNumber of qubits = ' + str(qubits))
    if qubits != 1:
        file_text.write('\nEntanglement = ' + entanglement)
    file_text.write('\nNumber of layers = ' + str(layers))
    file_text.write('\nMinimization method = '+ method)
    file_text.write('\nRandom seed = '+ str(seed))
    if method == 'SGD':
        file_text.write('\nNumber of epochs = '+ str(epochs))
    file_text.write('\n\nBEST RESULT\n\n')
    file_text.write('\nTHETA = \n')
    file_text.write(str(theta))
    file_text.write('\nALPHA = \n')
    file_text.write(str(alpha))
    if chi == 'weighted_fidelity_chi':
        file_text.write('\nWEIGHTS = \n')
        file_text.write(str(weights))
    file_text.write('\nchi**2 = ' + str(chi_value))
    file_text.write('\nacc_train = ' + str(acc_train))
    file_text.write('\nacc_test = ' + str(acc_test))
    file_text.close()
    
    np.savetxt(foldname + '/' + name + '_theta.txt', theta.flatten())
    np.savetxt(foldname + '/' + name + '_alpha.txt', alpha.flatten())
    if chi == 'weighted_fidelity_chi':
        np.savetxt(foldname + '/' + name + '_weight.txt', weights.flatten())
        

def read_summary(chi, problem, qubits, entanglement, layers, method, name):
        
    """
    This function reads the files saved by write_summary and returns theta, alpha and weight parameters
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines'
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        
    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
    if chi == 'fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
        if problem in ['circle', '3 circles', 'wavy circles', 'wavy lines', 'non convex','crown','tricrown','squares']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
        elif problem == 'sphere': 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 3
        elif problem in ['hypersphere']: 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 6))
            dim = 4
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))
        return theta, alpha
    
    if chi == 'weighted_fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
        if problem in ['circle', '3 circles', 'wavy circles', 'wavy lines', 'non convex','crown','tricrown','squares']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
        elif problem == 'sphere': 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 3
        elif problem in ['hypersphere']: 
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 6))
            dim = 4
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))

        if problem in ['3 circles','wavy lines','squares']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((4, qubits))
        if problem in ['circle','wavy circle','sphere', 'non convex', 'crown', 'hypersphere']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((2, qubits))
        if problem in ['tricrown']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((3, qubits))
        return theta, alpha, weight
    
    
def write_epochs_file(chi, problem, qubits, entanglement, layers, method, name):
        
    """
    This function creates a text file for saving data only in the SGD_step_by_step function
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
    OUTPUT:
        -file_text: an object which is an open textfile ready to be used
    """
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    create_folder(foldname)
    filename = foldname + '/' + name + '_epochs.txt'
    file_text = open(filename,'w')
    return file_text
    
def write_epoch(file_text, epoch, theta, alpha, chi_value, acc_train, acc_test):
    """
    This function takes a text file and write information on it
    INPUT: 
        -file_text: an object which is an open textfile ready to be used, output of write_epochs_file
        -epoch: the number of epoch providing this information
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. Must be an array with shape (classes, qubits)
        -chi_value: Value of the cost function after minimization
        -acc_train: accuracy for the training set
        -acc_test: accuracy for the test set
    OUTPUT:
        -file_text: with more information on it
    """
    file_text.write('\n Epoch = ' + str(epoch))
    file_text.write('\nTHETA = \n')
    file_text.write(str(theta))
    file_text.write('\nALPHA = \n')
    file_text.write(str(alpha))
    file_text.write('\n chi**2 = \n')
    file_text.write(str(chi_value))
    file_text.write('\nacc_train = \n')
    file_text.write(str(acc_train))
    file_text.write('\nacc_test = \n')
    file_text.write(str(acc_test))
    
def close_epochs_file(file_text, best_epoch):
    """
    This function takes a text file and closes it
    INPUT: 
        -file_text: an object which is an open textfile ready to be used, output of write_epochs_file after write_epoch
        -best_epoch: the epoch with the best possible results
    OUTPUT:
        -file_text: closed
    """
    file_text.write('\n\n\nBest epoch = ' + str(best_epoch))
    file_text.close()
    
def write_epochs_error_rate(chi, problem, qubits, entanglement, layers, method, name, 
                      accs_train, accs_test):   
    """
    This function takes information from the SGD_step_by_step function and saves the accuracies for training and test sets. It is required for studying the overlearning
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -accs_train: list or array with the accuracies of the training set for all epochs
        -accs_test: list or array with the accuracies of the test set for all epochs
    OUTPUT:
        Two files with the error rates in them
    """
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method)
    create_folder(foldname)
    filename_train = foldname + '/' + name + '_train.txt'
    filename_test = foldname + '/' + name + '_test.txt'
    
    np.savetxt(filename_train, 1 - np.array(accs_train))
    np.savetxt(filename_test, 1 - np.array(accs_test))
    
def samples_paint(problem, settings, sol, foldname, filename, bw):
    """
    This function takes the problem and the points when they are already classified, and saves a picture of them
    INPUT: 
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -settings: parameters the function needs for drawing. Provided by problem_gen.problem_gen
        -sol: solutions of the points alreafy classified
        -foldname : name of the folder where we store results
        -filename: name of the files we will produce
        -bw: black and white, True/False
    OUTPUT:
        a file with the points and their classes, and whether they are right or wrong
    """
    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5,vmax=np.max(sol[:,-3]) + .5)
    
        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)
        
    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.1,vmax=np.max(sol[:,-3]) + .1)
    
        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)

    fig, axs = plt.subplots(ncols = 2, figsize=(10,5))
    ax = axs[0]
    if problem in ['circle', '3 circles', 'crown', 'tricrown']:
        centers, radii = settings
        phi = np.linspace(0, 2*np.pi, 100)
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            ax.add_artist(ca)
    elif problem == 'wavy circle':
        centers, radii, wave, freq = settings
        phi = np.linspace(0, 2*np.pi, 1000)
        for (c,r, w, f) in zip(centers, radii, wave, freq):
            ax.plot(c[0] + r*(1 + w * np.cos(f * phi)) * np.cos(phi),
                    c[1] + r*(1 + w * np.cos(f * phi)) * np.sin(phi),
                    'k-')
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        ax.plot(s, -s + np.sin(freq * np.pi * s), 'k-')
    elif problem == 'squares':
        freq = settings
        s = np.linspace(-1,1,10)
        ax.plot(s, np.zeros(10), 'k-')
        ax.plot(np.zeros(10), s, 'k-')
        
    elif problem == 'non convex':
        freq, x_val, sin_val = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(-x_val * s + sin_val * np.sin(freq * np.pi * s), -1, 1), 'k-')

    ax.scatter(sol[:,0], sol[:,1], c=sol[:,-2], cmap = colors_classes, s=2, norm=norm_class)
    
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.margins(0)
    ax.axis('equal')
    
    bx = axs[1]    
    bx.scatter(sol[:,0], sol[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)  
    if problem in ['circle', '3 circles', 'crown', 'tricrown']:
        centers, radii = settings
        phi = np.linspace(0, 2*np.pi, 100)
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            bx.add_artist(ca)
    elif problem == 'wavy circle':
        centers, radii, wave, freq = settings
        phi = np.linspace(0, 2*np.pi, 1000)
        bx.plot(c[0] + r*(1 + wave * np.cos(freq * phi)) * np.cos(phi),
                c[1] + r*(1 + wave * np.cos(freq * phi)) * np.sin(phi),
                'k-')
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        bx.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'squares':
        freq = settings
        s = np.linspace(-1,1,10)
        bx.plot(s, np.zeros(10), 'k-')
        bx.plot(np.zeros(10), s, 'k-')
        
    elif problem == 'non convex':
        freq, x_val, sin_val = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(-x_val * s + sin_val * np.sin(freq * np.pi * s), -1, 1), 'k-')

    
    bx.set_xlabel('x', fontsize=16)
    bx.tick_params(axis='x', labelsize = 16)
    bx.tick_params(axis='y', labelsize=0)
    bx.set_xlim([-1, 1])
    bx.set_ylim([-1, 1])
    bx.margins(0)
    bx.axis('equal')
    
    fig.savefig(foldname + '/' + filename)
    plt.close('all')
    
'''
def samples_paint_worldmap(problem, settings, sol, foldname, filename, bw):
    """
    This function takes the problem and the points when they are already classified, and saves a picture of them
    INPUT: 
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -settings: parameters the function needs for drawing. Provided by problem_gen.problem_gen
        -sol: solutions of the points alreafy classified
        -foldname : name of the folder where we store results
        -filename: name of the files we will produce
        -bw: black and white, True/False
    OUTPUT:
        a file with the points and their classes, and whether they are right or wrong
    """
    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5,vmax=np.max(sol[:,-3]) + .5)
    
        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)
        
    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.5,vmax=np.max(sol[:,-3]) + .5)
    
        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)

    fig, axs = plt.subplots(nrows = 3, figsize=(5,15))
    
    line1 = _winkel_map((np.linspace(-np.pi,np.pi), np.zeros(50)))
    line2 = _winkel_map((np.linspace(-np.pi,np.pi), np.ones(50)))
    line3 = _winkel_map((np.linspace(-np.pi,np.pi), -np.ones(50)))
    line4 = _winkel_map((np.zeros(50), np.linspace(-np.pi/2,.5*np.pi)))
    line5 = _winkel_map((np.pi*np.ones(50), np.linspace(-np.pi/2,.5*np.pi)))
    line6 = _winkel_map((-np.pi*np.ones(50), np.linspace(-np.pi/2,.5*np.pi)))
    ax = axs[0]
    ax.plot(line1[0], line1[1], 'k')
    ax.plot(line2[0], line2[1], 'k')
    ax.plot(line3[0], line3[1], 'k')
    ax.plot(line4[0], line4[1], 'k')
    ax.plot(line5[0], line5[1], 'k')
    ax.plot(line6[0], line6[1], 'k')

    X = np.empty((len(sol), 2))
    for i,s in enumerate(sol):
        mapped = _winkel_map(s[:2])
        X[i] = mapped

    ax.scatter(X[:,0], X[:,1], c=sol[:,-3], cmap = colors_classes, s=2, norm=norm_class)
    
    #ax.set_xlabel('x', fontsize=16)
    #ax.set_ylabel('y', fontsize=16)
    #ax.tick_params(axis='both',labelsize=16)
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    #ax.margins(0)
    #ax.axis('equal')
    
    bx = axs[1]    
    bx.scatter(X[:,0], X[:,1], c=sol[:,-2], cmap = colors_classes, s=2, norm=norm_class)  
    
    cx = axs[2]    
    cx.scatter(X[:,0], X[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)  
    
    #bx.set_xlabel('x', fontsize=16)
    #bx.tick_params(axis='x', labelsize = 16)
    #bx.tick_params(axis='y', labelsize=0)
    #bx.set_xlim([-1, 1])
    #bx.set_ylim([-1, 1])
    #bx.margins(0)
    #bx.axis('equal')
    
    fig.savefig(foldname + '/' + filename + '_worldmap')
    plt.close('all')
    
def _winkel_map(angles):
    
    alpha = np.arccos(np.cos(angles[1])*np.cos(angles[0] / 2))
    x = .5 * (angles[0] * 180 / np.pi + 2 * np.cos(angles[1] * np.sin(.5 * angles[0])) / np.sinc(alpha / np.pi))
    y = .5 * (angles[1] * 180 / np.pi + np.sin(angles[1])/np.sinc(alpha/np.pi))
    
    return np.array([x,y])
'''
    
def create_folder(directory): 
    """
    Auxiliar function for creating directories with name directory
    
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def name_folder(chi, problem, qubits, entanglement, layers, method):
    """
    This function takes information from the SGD_step_by_step function and saves the accuracies for training and test sets. It is required for studying the overlearning
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -accs_train: list or array with the accuracies of the training set for all epochs
        -accs_test: list or array with the accuracies of the test set for all epochs
    OUTPUT:
        -foldname: A name for a folder
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
    foldname = chi + '/'
    problem = problem.replace(' ', '_')
    foldname += problem + '/'
    foldname += str(qubits) + '_qubits/'
    if qubits != 1: 
        if entanglement.lower()[0] == 'y':
            foldname += 'entangled/'
        if entanglement.lower()[0] == 'n':
            foldname += 'not_entangled/'
            
    foldname += str(layers) + '_layers/'
    foldname += method
    
    return foldname
    



